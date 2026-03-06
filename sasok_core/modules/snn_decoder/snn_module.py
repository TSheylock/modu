"""
SNN Emotion Decoder — Спайковая нейронная сеть для декодирования эмоций.

Третье поколение нейронных сетей с биологически правдоподобными
моделями нейронов (Leaky Integrate-and-Fire). Используется для
высокоточного декодирования эмоций из мультимодальных сигналов.

Архитектура:
  Input Layer  → Кодирование сигналов в спайковые последовательности
  Hidden Layer → LIF нейроны с латеральным торможением
  Output Layer → Декодирование спайков в эмоциональные категории

Преимущества SNN перед классическими ANN:
  - Энергоэффективность (обработка только при наличии спайков)
  - Временная динамика (информация в тайминге спайков)
  - Биологическая правдоподобность для эмоционального моделирования

Автор: Teymur Safiulov / SASOK v0.1.0
"""
import json
import time
import asyncio
import math
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from core.base_module import BaseModule


@dataclass
class LIFNeuron:
    """
    Leaky Integrate-and-Fire нейрон — биологически правдоподобная модель.

    Динамика мембранного потенциала:
      dV/dt = -(V - V_rest) / tau + I(t) / C

    Спайк генерируется когда V >= V_threshold,
    после чего V сбрасывается до V_reset на рефрактерный период.
    """
    neuron_id: str
    v_rest: float = -70.0       # мВ — потенциал покоя
    v_threshold: float = -55.0  # мВ — порог генерации спайка
    v_reset: float = -75.0      # мВ — потенциал после спайка
    tau: float = 20.0           # мс — постоянная времени мембраны
    r_membrane: float = 10.0    # МОм — сопротивление мембраны
    refractory_period: float = 2.0  # мс

    # Текущее состояние
    v_membrane: float = -70.0   # текущий мембранный потенциал
    last_spike_time: float = -999.0
    spike_count: int = 0
    is_refractory: bool = False

    # Входные веса
    input_weights: Dict[str, float] = field(default_factory=dict)

    def step(self, current_time: float, dt: float, input_current: float) -> bool:
        """
        Один шаг симуляции LIF нейрона.

        Args:
            current_time: Текущее время симуляции (мс)
            dt: Шаг времени (мс)
            input_current: Входной ток (нА)

        Returns:
            True если произошёл спайк
        """
        # Рефрактерный период
        if self.is_refractory:
            if current_time - self.last_spike_time >= self.refractory_period:
                self.is_refractory = False
            else:
                return False

        # Leaky integration
        dv = (-(self.v_membrane - self.v_rest) + self.r_membrane * input_current) / self.tau * dt
        self.v_membrane += dv

        # Проверка порога
        if self.v_membrane >= self.v_threshold:
            self.v_membrane = self.v_reset
            self.last_spike_time = current_time
            self.spike_count += 1
            self.is_refractory = True
            return True

        return False

    def reset(self):
        """Сброс нейрона в начальное состояние."""
        self.v_membrane = self.v_rest
        self.last_spike_time = -999.0
        self.spike_count = 0
        self.is_refractory = False


@dataclass
class SynapticConnection:
    """Синаптическое соединение между нейронами."""
    source_id: str
    target_id: str
    weight: float = 0.5
    delay: float = 1.0  # мс — синаптическая задержка
    plasticity: float = 0.01  # скорость STDP обучения


class SpikingNeuralNetwork:
    """
    Спайковая нейронная сеть для декодирования эмоций.

    3-слойная архитектура:
      - Input (6 нейронов): valence+, valence-, arousal+, arousal-, dominance+, dominance-
      - Hidden (12 нейронов): с латеральным торможением
      - Output (7 нейронов): joy, sadness, anger, fear, surprise, disgust, neutral
    """

    EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

    def __init__(self):
        self.neurons: Dict[str, LIFNeuron] = {}
        self.connections: List[SynapticConnection] = []
        self.spike_log: deque = deque(maxlen=1000)
        self._build_network()

    def _build_network(self):
        """Построение 3-слойной SNN."""
        # Input layer (6 нейронов — для положительных и отрицательных VAD)
        input_ids = ["in_val+", "in_val-", "in_aro+", "in_aro-", "in_dom+", "in_dom-"]
        for nid in input_ids:
            self.neurons[nid] = LIFNeuron(
                neuron_id=nid,
                v_threshold=-50.0,  # Более чувствительные входные нейроны
                tau=15.0
            )

        # Hidden layer (12 нейронов)
        hidden_ids = [f"hid_{i}" for i in range(12)]
        for nid in hidden_ids:
            self.neurons[nid] = LIFNeuron(
                neuron_id=nid,
                tau=20.0,
                v_threshold=-55.0
            )

        # Output layer (7 нейронов — по одному на эмоцию)
        for emotion in self.EMOTIONS:
            nid = f"out_{emotion}"
            self.neurons[nid] = LIFNeuron(
                neuron_id=nid,
                tau=25.0,
                v_threshold=-52.0  # Чуть ниже порог для более чётких решений
            )

        # Connections: input → hidden
        for in_id in input_ids:
            for hid_id in hidden_ids:
                weight = random.gauss(0.5, 0.15)
                weight = max(0.05, min(1.0, weight))
                self.connections.append(SynapticConnection(
                    source_id=in_id, target_id=hid_id,
                    weight=weight, delay=1.0 + random.random()
                ))

        # Connections: hidden → output
        for hid_id in hidden_ids:
            for emotion in self.EMOTIONS:
                out_id = f"out_{emotion}"
                weight = random.gauss(0.4, 0.2)
                weight = max(0.05, min(1.0, weight))
                self.connections.append(SynapticConnection(
                    source_id=hid_id, target_id=out_id,
                    weight=weight, delay=1.5 + random.random()
                ))

        # Lateral inhibition в hidden layer
        for i, hid_i in enumerate(hidden_ids):
            for j, hid_j in enumerate(hidden_ids):
                if i != j:
                    self.connections.append(SynapticConnection(
                        source_id=hid_i, target_id=hid_j,
                        weight=-0.15,  # Тормозящие связи
                        delay=0.5
                    ))

        # Предустановленные паттерны (какие входы активируют какие выходы)
        self._set_emotion_patterns()

    def _set_emotion_patterns(self):
        """Настройка весов для известных эмоциональных паттернов."""
        # joy: высокий valence+, средний arousal+
        self._strengthen_path(["in_val+"], "out_joy", boost=0.4)
        self._strengthen_path(["in_aro+"], "out_joy", boost=0.2)

        # sadness: высокий valence-, низкий arousal
        self._strengthen_path(["in_val-"], "out_sadness", boost=0.4)
        self._strengthen_path(["in_aro-"], "out_sadness", boost=0.3)

        # anger: высокий valence-, высокий arousal+, высокий dominance+
        self._strengthen_path(["in_val-"], "out_anger", boost=0.3)
        self._strengthen_path(["in_aro+"], "out_anger", boost=0.35)
        self._strengthen_path(["in_dom+"], "out_anger", boost=0.3)

        # fear: высокий valence-, высокий arousal+, низкий dominance
        self._strengthen_path(["in_val-"], "out_fear", boost=0.3)
        self._strengthen_path(["in_aro+"], "out_fear", boost=0.3)
        self._strengthen_path(["in_dom-"], "out_fear", boost=0.35)

        # surprise: высокий arousal+
        self._strengthen_path(["in_aro+"], "out_surprise", boost=0.5)

        # disgust: valence-, moderate arousal
        self._strengthen_path(["in_val-"], "out_disgust", boost=0.4)

        # neutral: low все
        self._strengthen_path(["in_aro-", "in_dom-"], "out_neutral", boost=0.3)

    def _strengthen_path(self, input_ids: List[str], output_id: str, boost: float):
        """Усиление путей от входов к выходу через скрытый слой."""
        for conn in self.connections:
            if conn.source_id in input_ids and conn.target_id.startswith("hid_"):
                # Усиление input→hidden для этих входов
                for conn2 in self.connections:
                    if conn2.source_id == conn.target_id and conn2.target_id == output_id:
                        conn2.weight = min(1.0, conn2.weight + boost)

    def encode_input(self, valence: float, arousal: float, dominance: float) -> Dict[str, float]:
        """
        Rate coding: преобразование VAD в входные токи.

        Биполярное кодирование: каждый параметр имеет два нейрона
        (положительный и отрицательный), ток пропорционален значению.
        """
        currents = {}

        # Valence: -1..1 → split into positive and negative
        currents["in_val+"] = max(0, valence) * 3.0  # нА
        currents["in_val-"] = max(0, -valence) * 3.0

        # Arousal: 0..1
        currents["in_aro+"] = arousal * 2.5
        currents["in_aro-"] = (1.0 - arousal) * 2.5

        # Dominance: 0..1
        currents["in_dom+"] = dominance * 2.0
        currents["in_dom-"] = (1.0 - dominance) * 2.0

        return currents

    def simulate(
        self,
        input_currents: Dict[str, float],
        duration_ms: float = 100.0,
        dt: float = 0.5
    ) -> Dict[str, Any]:
        """
        Запуск симуляции SNN.

        Args:
            input_currents: Входные токи для input нейронов
            duration_ms: Длительность симуляции в мс
            dt: Шаг времени

        Returns:
            Результат декодирования
        """
        # Reset всех нейронов
        for neuron in self.neurons.values():
            neuron.reset()

        # Spike trains
        spike_trains: Dict[str, List[float]] = {nid: [] for nid in self.neurons}

        # Симуляция
        steps = int(duration_ms / dt)
        for step in range(steps):
            t = step * dt
            pending_spikes: Dict[str, float] = {}  # target → accumulated current

            # Обработка каждого нейрона
            for nid, neuron in self.neurons.items():
                # Входной ток
                i_input = input_currents.get(nid, 0.0)

                # Синаптический ток от пришедших спайков
                i_synaptic = pending_spikes.get(nid, 0.0)

                # Шаг LIF
                spiked = neuron.step(t, dt, i_input + i_synaptic)

                if spiked:
                    spike_trains[nid].append(t)
                    # Распространение спайка по синапсам
                    for conn in self.connections:
                        if conn.source_id == nid:
                            target_current = pending_spikes.get(conn.target_id, 0.0)
                            pending_spikes[conn.target_id] = target_current + conn.weight * 2.0

        # Декодирование: подсчёт спайков output нейронов
        output_spikes = {}
        for emotion in self.EMOTIONS:
            nid = f"out_{emotion}"
            output_spikes[emotion] = len(spike_trains[nid])

        total_output_spikes = sum(output_spikes.values())

        # Нормализация в вероятности
        if total_output_spikes > 0:
            probabilities = {
                e: count / total_output_spikes
                for e, count in output_spikes.items()
            }
        else:
            probabilities = {e: 1.0 / len(self.EMOTIONS) for e in self.EMOTIONS}

        # Определение доминирующей эмоции
        dominant = max(probabilities, key=probabilities.get)
        dominant_confidence = probabilities[dominant]

        # Подсчёт общей статистики
        total_spikes = sum(len(train) for train in spike_trains.values())
        hidden_spikes = sum(
            len(spike_trains[f"hid_{i}"]) for i in range(12)
        )

        return {
            "dominant_emotion": dominant,
            "confidence": round(dominant_confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "output_spike_counts": output_spikes,
            "total_spikes": total_spikes,
            "hidden_layer_spikes": hidden_spikes,
            "simulation_duration_ms": duration_ms,
            "dt_ms": dt,
            "energy_efficiency": round(total_spikes / steps, 4)  # спайков на шаг
        }

    def apply_stdp(self, pre_spike_time: float, post_spike_time: float,
                   connection: SynapticConnection):
        """
        Spike-Timing Dependent Plasticity (STDP).

        Если pre спайкнул ДО post → усиление (LTP).
        Если pre спайкнул ПОСЛЕ post → ослабление (LTD).
        """
        dt = post_spike_time - pre_spike_time
        tau_plus = 20.0  # мс
        tau_minus = 20.0

        if dt > 0:
            # LTP: pre before post
            dw = connection.plasticity * math.exp(-dt / tau_plus)
        else:
            # LTD: post before pre
            dw = -connection.plasticity * 0.5 * math.exp(dt / tau_minus)

        connection.weight = max(-0.5, min(1.0, connection.weight + dw))


class SNNDecoderModule(BaseModule):
    """
    Модуль SNN-декодирования эмоций для SASOK.

    Использует спайковую нейронную сеть (LIF нейроны) для биологически
    правдоподобного декодирования эмоций из VAD-вектора XoCore.
    Работает параллельно с основным emotion модулем для перекрёстной
    верификации.
    """

    async def initialize(self):
        self.logger.info("Инициализация SNN Emotion Decoder...")

        self.snn = SpikingNeuralNetwork()
        self._decode_history: deque = deque(maxlen=100)

        self.state = {
            "active": False,
            "decodes_total": 0,
            "last_decode": None,
            "dominant_emotion": "neutral",
            "network_stats": {
                "input_neurons": 6,
                "hidden_neurons": 12,
                "output_neurons": 7,
                "total_connections": len(self.snn.connections),
                "neuron_model": "Leaky Integrate-and-Fire"
            }
        }

        self.logger.info(
            f"SNN инициализирована: {len(self.snn.neurons)} нейронов, "
            f"{len(self.snn.connections)} синапсов, "
            f"модель: LIF"
        )

    async def activate(self):
        if self.active:
            return
        self.logger.info("Активация SNN Emotion Decoder...")

        await self.subscribe("emotion.state_changed", self._on_emotion_state)
        await self.subscribe("snn.decode_request", self._on_decode_request)

        self.active = True
        await self.update_state({"active": True})
        self.logger.info("SNN Emotion Decoder активирован")

    async def deactivate(self):
        if not self.active:
            return
        self.logger.info("Деактивация SNN Emotion Decoder...")

        for sub in self.subscriptions:
            await sub.unsubscribe()
        self.subscriptions = []

        self.active = False
        await self.update_state({"active": False})
        self.logger.info("SNN Emotion Decoder деактивирован")

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.active:
            return {"error": "Module inactive"}

        action = data.get("action", "decode")

        if action == "decode":
            return await self._decode_emotion(
                valence=data.get("valence", 0.0),
                arousal=data.get("arousal", 0.5),
                dominance=data.get("dominance", 0.5),
                simulation_ms=data.get("simulation_ms", 100.0)
            )
        elif action == "network_info":
            return {
                "neurons": len(self.snn.neurons),
                "connections": len(self.snn.connections),
                "emotions": self.snn.EMOTIONS,
                "model": "Leaky Integrate-and-Fire",
                "architecture": "3-layer (6-12-7) with lateral inhibition"
            }
        elif action == "history":
            return {"history": list(self._decode_history)}
        elif action == "status":
            return self.state
        else:
            return {"error": f"Unknown action: {action}"}

    async def _decode_emotion(
        self,
        valence: float,
        arousal: float,
        dominance: float,
        simulation_ms: float = 100.0
    ) -> Dict[str, Any]:
        """Декодирование эмоции через SNN."""
        # Кодирование входов
        input_currents = self.snn.encode_input(valence, arousal, dominance)

        # Симуляция
        result = self.snn.simulate(input_currents, simulation_ms)

        # Обновление состояния
        self.state["decodes_total"] = self.state.get("decodes_total", 0) + 1
        self.state["last_decode"] = {
            "emotion": result["dominant_emotion"],
            "confidence": result["confidence"],
            "timestamp": time.time()
        }
        self.state["dominant_emotion"] = result["dominant_emotion"]

        # Логирование
        decode_record = {
            "input": {"valence": valence, "arousal": arousal, "dominance": dominance},
            "output": result["dominant_emotion"],
            "confidence": result["confidence"],
            "timestamp": time.time()
        }
        self._decode_history.append(decode_record)

        self.logger.info(
            f"🧠 SNN decode: VAD({valence:.2f}, {arousal:.2f}, {dominance:.2f}) "
            f"→ {result['dominant_emotion']} ({result['confidence']:.2%}), "
            f"{result['total_spikes']} spikes"
        )

        # Публикация результата
        await self.publish(
            "snn.emotion_decoded",
            json.dumps({
                "emotion": result["dominant_emotion"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "total_spikes": result["total_spikes"],
                "source": "snn_decoder"
            }).encode("utf-8")
        )

        return {
            "success": True,
            **result
        }

    async def _on_emotion_state(self, msg):
        """Декодирование при получении нового эмоционального состояния."""
        try:
            data = json.loads(msg.data.decode())
            await self._decode_emotion(
                valence=data.get("valence", 0.0),
                arousal=data.get("arousal", 0.5),
                dominance=data.get("dominance", 0.5)
            )
        except Exception as e:
            self.logger.error(f"Ошибка декодирования эмоции: {e}")

    async def _on_decode_request(self, msg):
        try:
            data = json.loads(msg.data.decode())
            result = await self.process(data)
            await self.publish(
                "snn.decode_result",
                json.dumps(result).encode("utf-8")
            )
        except Exception as e:
            self.logger.error(f"Ошибка обработки запроса декодирования: {e}")
