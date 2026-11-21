# -*- coding: utf-8 -*-
"""
red-xonidu v3.1 - Agente con Razonamiento Mejorado
Incluye: multi-hop reasoning, mejor formateo, memoria de sesión
"""

import os
import re
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta

# Cargar configuración
CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
import yaml
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

logger = logging.getLogger(__name__)


class ToolType(Enum):
    WIKIPEDIA = "wikipedia"
    ARXIV = "arxiv"
    WEATHER = "weather"
    NEWS = "news"
    WEB = "web"
    NONE = "none"


@dataclass
class ToolCall:
    tool: ToolType
    query: str
    raw_response: str = ""
    reasoning: str = ""


@dataclass
class AgentResponse:
    final_answer: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    confidence: float = 0.0
    success: bool = True
    error: str = ""


class IntentClassifier:
    """Clasificador de intención híbrido: keywords + modelo ligero"""

    PATTERNS = {
        ToolType.WEATHER: [
            (r'\b(clima|tiempo|temperatura|llover|lluvia|pronóstico|pronostico)\b', 3),
            (r'\b(hace|está|va a)\s+(calor|frío|sol|nublado|lluvioso|buen|mal)\b', 2),
        ],
        ToolType.ARXIV: [
            (r'\b(paper|papers|artículo|articulo|publicación|investigación|estudio)\b', 3),
            (r'\b(arxiv|publicado|reciente|científico|preprint)\b', 2),
            (r'\bbusca.*paper\b', 3),
        ],
        ToolType.NEWS: [
            (r'\b(noticia|noticias|actualidad|última hora|reciente|hoy)\b', 3),
            (r'\b(verifica|fact.?check|comprueba|confirmar)\b', 2),
        ],
        ToolType.WIKIPEDIA: [
            (r'^(qué|que|quién|quien|cuándo|cuando|dónde|donde|cómo|como)\s+(es|era|fue|significa|significan)', 3),
            (r'\b(definición|definicion|significado|concepto|historia|biografía|biografia|qué es|que es)\b', 3),
            (r'\b(explícame|explicame|dime|cuéntame|cuentame)\b', 2),  # Instrucciones explicativas
            (r'\b(wikipedia|wiki|enciclopedia)\b', 2),
            (r'\b(compara|comparar|diferencia|versus|vs)\b', 2),  # Comparaciones → Wikipedia
        ],
        ToolType.WEB: [
            (r'\b(cómo|como)\s+(hago|hacer|tramitar|sacar|obtener|puedo|instalar|configurar|descargar)\b', 3),
            (r'\b(dónde|donde)\s+(está|queda|puedo|encuentro|encuentra)\b', 3),
            (r'\b(trámite|tramite|gestión|solicitar)\b', 2),
            (r'\b(busca|buscar|encontrar|información|informacion)\b', 1),
        ],
    }

    @classmethod
    def classify(cls, query: str) -> tuple[ToolType, float]:
        query_lower = query.lower()

        scores = {}
        for tool_type, patterns in cls.PATTERNS.items():
            score = 0
            for pattern, weight in patterns:
                if re.search(pattern, query_lower):
                    score += weight
            scores[tool_type] = score

        best = max(scores, key=scores.get)
        max_score = scores[best]
        confidence = min(max_score / 5.0, 1.0) if max_score > 0 else 0.0

        # Si no hay coincidencia clara, no forzar a una herramienta: devolver NONE
        # para que el agente responda sin depender de APIs externas (clave en 1 GB RAM).
        if max_score == 0:
            return ToolType.NONE, 0.0

        return best, confidence


class ReasoningEngine:
    """Motor de razonamiento simple para descomponer consultas complejas"""

    @staticmethod
    def decompose_query(query: str) -> List[str]:
        """Descompone una consulta compleja en sub-consultas"""
        # Detectar consultas compuestas
        sub_queries = []
        query_lower = query.lower()

        # 1. Primero: patrones de comparación explícitos
        if ' compar' in query_lower:
            # "compara X con Y", "compara X y Y", "comparativa X vs Y"
            match = re.search(r'compar[a-z]*\s+(.+?)\s+(con|y|vs|versus)\s+(.+)', query, re.IGNORECASE)
            if match:
                sub_queries = [match.group(1).strip(), match.group(3).strip()]
            else:
                # Fallback: buscar dos entidades separadas por "y" después de compara
                match = re.search(r'compar[a-z]*\s+(.+)', query, re.IGNORECASE)
                if match:
                    rest = match.group(1)
                    if ' y ' in rest:
                        sub_queries = [p.strip() for p in rest.split(' y ')]
                    elif ' vs ' in rest:
                        sub_queries = [p.strip() for p in rest.split(' vs ')]

        # 2. Segundo: vs/versus
        elif ' vs ' in query_lower or ' versus ' in query_lower:
            parts = re.split(r'\s+(vs|versus)\s+', query, flags=re.IGNORECASE)
            sub_queries = [p.strip() for p in parts]

        # 3. Tercero: "y" genérico - SOLO si no es comparación
        elif ' y ' in query_lower or ' e ' in query_lower:
            # Evitar dividir:
            # - "cómo hago X y Y" (una sola pregunta con dos acciones)
            # - "qué es X y Y" / "qué son X y Y" (definición de conceptos relacionados)
            # - "dónde está X y Y" (ubicación)
            # - "explícame X y hazme Y" / "explícame X y dime Y" (instrucción compuesta)
            # Patrones que indican pregunta única con "y" conectando conceptos
            avoid_split = (
                re.search(r'(cómo|como|dónde|donde)\s+\w+\s+y\s+\w+', query_lower) or
                re.search(r'(qué|que)\s+(es|son|está|estan)\s+.+\s+y\s+\w+', query_lower) or
                re.search(r'(explícame|explicame|dime|cuéntame|cuentame)\s+.+\s+y\s+(hazme|dime|cuéntame|cuentame|explícame|explicame)', query_lower)
            )
            if not avoid_split:
                parts = re.split(r'\s+(y|e)\s+', query)
                sub_queries = [p.strip() for p in parts if p not in ['y', 'e']]

        # Filtrar sub-queries muy cortas (menos de 2 palabras)
        sub_queries = [q for q in sub_queries if len(q.split()) >= 2]

        return sub_queries if len(sub_queries) > 1 else [query]

    @staticmethod
    def needs_multiple_tools(query: str) -> bool:
        """Detecta si la consulta necesita múltiples herramientas"""
        indicators = [
            'y también', 'además', 'también', 'y luego', 'después',
            'compara', 'comparar', 'vs', 'versus', 'diferencia'
        ]
        return any(ind in query.lower() for ind in indicators)


class ToolExecutor:
    """Ejecuta las llamadas a APIs externas con cache simple"""

    def __init__(self, low_ram: bool = False):
        self.session = None
        self._cache = {}
        self.low_ram = low_ram
        self._init_session()

    def _init_session(self):
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'red-xonidu/3.1 (+https://github.com/xonidu/red-xonidu)'
            })
        except ImportError:
            logger.warning("requests no instalado")

    def _timeout(self, default: int) -> int:
        return 8 if self.low_ram else default

    def execute(self, tool_call: ToolCall) -> str:
        if not self.session:
            return "Error: requests no disponible"

        cache_key = f"{tool_call.tool.value}:{tool_call.query}"
        if cache_key in self._cache:
            logger.info(f"Cache hit: {cache_key}")
            return self._cache[cache_key]

        tool = tool_call.tool
        query = tool_call.query

        try:
            if tool == ToolType.WIKIPEDIA:
                result = self._search_wikipedia(query)
            elif tool == ToolType.ARXIV:
                result = self._search_arxiv(query)
            elif tool == ToolType.WEATHER:
                result = self._search_weather(query)
            elif tool == ToolType.NEWS:
                result = self._search_news(query)
            elif tool == ToolType.WEB:
                result = self._search_web(query)
            else:
                result = "Herramienta no reconocida"
        except Exception as e:
            logger.error(f"Error en {tool.value}: {e}")
            result = f"Error: {str(e)}"

        # Cache por 5 minutos
        self._cache[cache_key] = result
        return result

    def _search_wikipedia(self, query: str) -> str:
        import requests
        clean_query = re.sub(r'^(qué|que|quién|quien|cuándo|cuando|dónde|donde|cómo|como)\s+(es|era|fue|significa|significan)\s+', '', query, flags=re.IGNORECASE)
        clean_query = re.sub(r'^(explícame|explicame|define|definición|definicion)\s+', '', clean_query, flags=re.IGNORECASE)
        clean_query = re.split(r'\s+(de forma|para|y hazme|con|sobre|acerca)\b', clean_query, flags=re.IGNORECASE)[0]

        params = {
            'action': 'query', 'format': 'json', 'list': 'search',
            'srsearch': clean_query.strip(), 'srlimit': 3, 'srprop': 'snippet'
        }
        r = self.session.get(CONFIG['apis']['wikipedia']['base_url'], params=params, timeout=self._timeout(10))
        data = r.json()

        results = []
        for item in data.get('query', {}).get('search', []):
            title = item['title']
            snippet = item['snippet'].replace('<span class="searchmatch">', '').replace('</span>', '')
            results.append(f"📖 **{title}**: {snippet}")

        if not results:
            return "Sin resultados en Wikipedia"
        return "\n\n".join(results)

    def _search_arxiv(self, query: str) -> str:
        import requests
        import xml.etree.ElementTree as ET

        clean_query = re.sub(r'\b(busca|buscar|paper|papers|artículo|articulo|sobre|de|en|reciente|últimos|ultimos)\b', '', query, flags=re.IGNORECASE).strip()

        params = {
            'search_query': f'all:"{clean_query}"',
            'start': 0, 'max_results': 5,
            'sortBy': 'relevance', 'sortOrder': 'descending'
        }
        r = self.session.get(CONFIG['apis']['arxiv']['base_url'], params=params, timeout=self._timeout(15))
        root = ET.fromstring(r.content)

        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        results = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip()
            summary = entry.find('atom:summary', ns).text.strip()[:300]
            link = entry.find('atom:id', ns).text.strip()
            results.append(f"📄 **{title}**\n{summary}...\n🔗 {link}")

        if not results:
            return "Sin papers en ArXiv"
        return "\n\n---\n\n".join(results)

    def _search_weather(self, query: str) -> str:
        import requests

        api_key = os.getenv(CONFIG['apis']['weather']['api_key_env'])
        if not api_key:
            return "⚠️ API key de OpenWeatherMap no configurada (variable OPENWEATHER_API_KEY)"

        city = re.sub(r'\b(clima|tiempo|temperatura|pronóstico|pronostico|hace|está|va a|en|de)\b', '', query, flags=re.IGNORECASE).strip()
        if not city:
            city = query

        params = {'q': city, 'appid': api_key, 'units': 'metric', 'lang': 'es'}
        r = self.session.get(f"{CONFIG['apis']['weather']['base_url']}/weather", params=params, timeout=self._timeout(10))

        if r.status_code != 200:
            return f"❌ Error: {r.json().get('message', 'Ciudad no encontrada')}"

        data = r.json()
        return (f"🌤️ **Clima en {data['name']}, {data['sys']['country']}**\n"
                f"• 🌡️ Temperatura: {data['main']['temp']}°C\n"
                f"• ☁️ Condición: {data['weather'][0]['description']}\n"
                f"• 💧 Humedad: {data['main']['humidity']}%\n"
                f"• 💨 Viento: {data['wind']['speed']} m/s")

    def _search_news(self, query: str) -> str:
        import requests

        api_key = os.getenv(CONFIG['apis']['news']['api_key_env'])
        if not api_key:
            return "⚠️ API key de GNews no configurada (variable GNEWS_API_KEY)"

        clean_query = re.sub(r'\b(noticia|noticias|busca|buscar|sobre|de|actualidad|verifica|fact.?check)\b', '', query, flags=re.IGNORECASE).strip()

        params = {'q': clean_query, 'token': api_key, 'lang': 'es', 'max': 5}
        r = self.session.get(f"{CONFIG['apis']['news']['base_url']}/search", params=params, timeout=self._timeout(10))

        if r.status_code != 200:
            return f"❌ Error noticias: {r.status_code}"

        data = r.json()
        articles = data.get('articles', [])

        if not articles:
            return "Sin noticias encontradas"

        results = []
        for a in articles[:3]:
            title = a.get('title', '')
            desc = a.get('description', '')[:200]
            source = a.get('source', {}).get('name', '')
            url = a.get('url', '')
            results.append(f"📰 **{title}** ({source})\n{desc}...\n🔗 {url}")

        return "\n\n---\n\n".join(results)

    def _search_web(self, query: str) -> str:
        import requests

        clean_query = re.sub(r'\b(busca|buscar|encontrar|información|informacion|trámite|tramite|cómo|como|dónde|donde)\b', '', query, flags=re.IGNORECASE).strip()

        # 1. DuckDuckGo Instant Answer
        params = {'q': clean_query, 'format': 'json', 'no_html': 1, 'skip_disambig': 1}
        r = self.session.get(CONFIG['apis']['duckduckgo']['base_url'], params=params, timeout=self._timeout(10))
        data = r.json()

        results = []
        if data.get('Abstract'):
            results.append(f"📋 {data['Abstract']}")
        if data.get('AbstractText'):
            results.append(f"📝 {data['AbstractText']}")

        for topic in data.get('RelatedTopics', [])[:3]:
            if isinstance(topic, dict) and 'Text' in topic:
                results.append(f"🔗 {topic['Text'][:200]}")

        # 2. Fallback a Wikipedia
        if not results:
            wiki_params = {
                'action': 'query', 'format': 'json', 'list': 'search',
                'srsearch': clean_query, 'srlimit': 3, 'srprop': 'snippet'
            }
            wiki_r = self.session.get(CONFIG['apis']['wikipedia']['base_url'], params=wiki_params, timeout=self._timeout(10))
            wiki_data = wiki_r.json()

            for item in wiki_data.get('query', {}).get('search', []):
                title = item['title']
                snippet = item['snippet'].replace('<span class="searchmatch">', '').replace('</span>', '')
                results.append(f"📖 **{title}**: {snippet}")

        if not results:
            return "Sin resultados en búsqueda web"
        return "\n\n".join(results)


class ResponseFormatter:
    """Formateador de respuestas con plantillas por tipo de herramienta"""

    TEMPLATES = {
        ToolType.WIKIPEDIA: "📚 **Definición/Concepto**\n\n{content}",
        ToolType.ARXIV: "📄 **Papers científicos encontrados**\n\n{content}",
        ToolType.WEATHER: "{content}",
        ToolType.NEWS: "📰 **Noticias recientes**\n\n{content}",
        ToolType.WEB: "🔍 **Resultados de búsqueda**\n\n{content}",
    }

    @classmethod
    def format(cls, user_query: str, tool_results: List[str], tool_calls: List[ToolCall]) -> str:
        if not tool_results:
            return "No encontré información sobre eso."

        # Si todas las herramientas son NONE, respuesta local sin API externa
        if all(tc.tool == ToolType.NONE for tc in tool_calls):
            return ("No tengo una respuesta precisa para eso sin consultar fuentes externas. "
                    "Prueba reformulando la pregunta o especifica el tema (definición, clima, noticias, paper).")

        # Si solo una herramienta, usar su template
        if len(tool_calls) == 1:
            template = cls.TEMPLATES.get(tool_calls[0].tool, "{content}")
            return template.format(content=tool_results[0])

        # Múltiples herramientas: combinar
        combined = []
        for tc, result in zip(tool_calls, tool_results):
            template = cls.TEMPLATES.get(tc.tool, "{content}")
            combined.append(template.format(content=result))

        return "\n\n---\n\n".join(combined)


class SessionMemory:
    """Memoria de sesión para continuidad conversacional"""

    def __init__(self, max_history: int = 10):
        self.history: List[Dict] = []
        self.max_history = max_history
        self.entities: Dict[str, List[str]] = {}  # Entidades mencionadas

    def add(self, role: str, content: str, tool_calls: List[ToolCall] = None):
        self.history.append({
            'role': role,
            'content': content,
            'tool_calls': [tc.tool.value for tc in (tool_calls or [])],
            'timestamp': datetime.now().isoformat()
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Extraer entidades simples
        self._extract_entities(content)

    def _extract_entities(self, text: str):
        # Palabras capitalizadas y términos técnicos
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for w in words:
            if len(w) > 2:
                self.entities.setdefault(w, []).append(text[:50])

    def get_context(self, max_items: int = 3) -> str:
        if not self.history:
            return ""
        recent = self.history[-max_items:]
        lines = []
        for h in recent:
            role = "Usuario" if h['role'] == 'user' else "Asistente"
            lines.append(f"{role}: {h['content'][:100]}")
        return "\n".join(lines)

    def get_relevant_entities(self, query: str) -> List[str]:
        query_words = set(query.lower().split())
        relevant = []
        for entity, contexts in self.entities.items():
            if entity.lower() in query.lower():
                relevant.append(entity)
        return relevant


class EnhancedAgent:
    """Agente mejorado con razonamiento multi-paso y memoria"""

    def __init__(self):
        self.classifier = IntentClassifier()
        self.reasoning = ReasoningEngine()
        self.tools = ToolExecutor(low_ram=self._low_ram())
        self.formatter = ResponseFormatter()
        self.memory = SessionMemory()
        self.max_hops = 2
        # Caché en disco con TTL para evitar peticiones repetidas (red lenta, 1 GB RAM)
        self.cache_dir = Path(__file__).parent.parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl_seconds = 300  # 5 minutos

    @staticmethod
    def _low_ram() -> bool:
        """Detecta si el equipo tiene poca RAM (ej. ASUS Eee 900 con 1 GB)"""
        try:
            import psutil
            return psutil.virtual_memory().total < 2 * 1024**3
        except Exception:
            try:
                with open('/proc/meminfo') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            kb = int(line.split()[1])
                            return kb < 2 * 1024**2
            except Exception:
                pass
        return False

    def _cache_path(self, cache_key: str) -> Path:
        safe = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{safe}.json"

    def _cache_get(self, cache_key: str) -> Optional[str]:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(data["ts"])
            if datetime.now() - ts > timedelta(seconds=self.cache_ttl_seconds):
                path.unlink(missing_ok=True)
                return None
            return data["value"]
        except Exception:
            return None

    def _cache_set(self, cache_key: str, value: str) -> None:
        path = self._cache_path(cache_key)
        try:
            path.write_text(
                json.dumps({"ts": datetime.now().isoformat(), "value": value},
                           ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"No se pudo escribir caché: {e}")

    def query(self, user_input: str) -> AgentResponse:
        # 1. Verificar si necesita descomposición
        sub_queries = self.reasoning.decompose_query(user_input)

        all_tool_calls = []
        all_results = []
        reasoning_chain = []

        # 2. Procesar cada sub-query
        for i, sub_q in enumerate(sub_queries):
            if i > 0:
                # Para sub-queries adicionales, añadir contexto previo
                context = self.memory.get_context()
                enhanced_q = f"{sub_q} (contexto: {context})"
            else:
                enhanced_q = sub_q

            # 3. Clasificar intención
            tool_type, confidence = self.classifier.classify(enhanced_q)
            reasoning_chain.append(f"Paso {i+1}: '{sub_q}' → {tool_type.value} (conf: {confidence:.2f})")

            # 4. Sin coincidencia clara: responder sin herramientas externas
            if tool_type == ToolType.NONE:
                tool_call = ToolCall(tool=tool_type, query=sub_q)
                tool_call.raw_response = "respuesta_local"
                all_tool_calls.append(tool_call)
                all_results.append("")
                reasoning_chain.append("  → Sin herramienta externa (respuesta local)")
                continue

            # 5. Preparar query limpia
            tool_query = self._extract_tool_query(enhanced_q, tool_type)
            tool_call = ToolCall(tool=tool_type, query=tool_query)
            tool_call.reasoning = reasoning_chain[-1]

            # 6. Caché en disco antes de pedir a la red
            cache_key = f"{tool_call.tool.value}:{tool_call.query}"
            cached = self._cache_get(cache_key)
            if cached is not None:
                result = cached
                reasoning_chain.append(f"  → Caché: {result[:80]}...")
            else:
                # 7. Ejecutar herramienta
                result = self.tools.execute(tool_call)
                tool_call.raw_response = result
                self._cache_set(cache_key, result)

            all_tool_calls.append(tool_call)
            all_results.append(result)
            reasoning_chain.append(f"  → Resultado: {result[:80]}...")

        # 8. Formatear respuesta final
        final_answer = self.formatter.format(user_input, all_results, all_tool_calls)

        # 9. Calcular confianza promedio
        avg_confidence = sum(self.classifier.classify(q)[1] for q in sub_queries) / len(sub_queries)

        # 10. Guardar en memoria
        self.memory.add('user', user_input, all_tool_calls)
        self.memory.add('assistant', final_answer, all_tool_calls)

        return AgentResponse(
            final_answer=final_answer,
            tool_calls=all_tool_calls,
            reasoning_chain=reasoning_chain,
            confidence=avg_confidence,
            success=True
        )

    def _extract_tool_query(self, user_input: str, tool_type: ToolType) -> str:
        if tool_type == ToolType.NONE:
            return ""
        if tool_type == ToolType.WEATHER:
            cleaned = re.sub(r'\b(clima|tiempo|temperatura|pronóstico|pronostico|hace|está|va a|en|de|el|la|cómo|como)\b', '', user_input, flags=re.IGNORECASE)
            # Limpiar puntuación y espacios extra
            cleaned = re.sub(r'[¿?¡!]', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            return cleaned.strip()
        elif tool_type == ToolType.ARXIV:
            return re.sub(r'\b(busca|buscar|paper|papers|artículo|articulo|sobre|de|en|reciente|últimos|ultimos|arxiv|científicos|cientificos|recientes)\b', '', user_input, flags=re.IGNORECASE).strip()
        elif tool_type == ToolType.NEWS:
            cleaned = re.sub(r'\b(noticia|noticias|busca|buscar|sobre|de|actualidad|verifica|fact.?check|recientes)\b', '', user_input, flags=re.IGNORECASE)
            cleaned = re.sub(r'[¿?¡!]', '', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            return cleaned.strip()
        elif tool_type == ToolType.WEB:
            cleaned = re.sub(r'^\b(cómo|como|dónde|donde|qué|que)\s+(hago|hacer|puedo|tramitar|sacar|obtener|encuentro|encuentra)\b\s*', '', user_input, flags=re.IGNORECASE)
            cleaned = re.sub(r'^\b(busca|buscar|encontrar|información|informacion|trámite|tramite)\b\s*', '', cleaned, flags=re.IGNORECASE)
            words = [w for w in cleaned.split() if len(w) > 3]
            return ' '.join(words[:5]) if words else cleaned.strip()
        else:  # WIKIPEDIA
            # Limpiar patrones de comparación
            cleaned = re.sub(r'^\b(compara|comparar|comparativa|diferencia|versus|vs)\b\s*', '', user_input, flags=re.IGNORECASE)
            # Limpiar " y " que queda de la comparación
            cleaned = re.sub(r'\s+y\s+', ' ', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'^(qué|que|quién|quien|cuándo|cuando|dónde|donde|cómo|como)\s+(es|era|fue|significa|significan)\s+', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'^(explícame|explicame|define|definición|definicion)\s+', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.split(r'\s+(de forma|para|y hazme|con|sobre|acerca)\b', cleaned, flags=re.IGNORECASE)[0]
            return cleaned.strip()

    def reset_conversation(self):
        self.memory = SessionMemory()


def create_agent() -> EnhancedAgent:
    return EnhancedAgent()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    agent = create_agent()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Pregunta: ")

    print(f"\n🤖 Procesando: {query}\n")
    response = agent.query(query)

    print(f"Respuesta:\n{response.final_answer}")
    print(f"\n--- Confianza: {response.confidence:.2f} ---")
    print(f"--- Razonamiento ---")
    for step in response.reasoning_chain:
        print(f"  {step}")
    print(f"\n--- Herramientas usadas: {len(response.tool_calls)} ---")
    for tc in response.tool_calls:
        print(f"  {tc.tool.value}: {tc.query}")