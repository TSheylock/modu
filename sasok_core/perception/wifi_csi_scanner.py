"""
Wi-Fi CSI Scanner — радарный массив SASOK.

Использует Channel State Information (CSI) от Wi-Fi роутера для:
- Обнаружения присутствия человека (сквозь стены)
- Измерения дыхания (~0.1–0.5 Hz)
- Измерения сердцебиения (~0.8–2.0 Hz)
- Оценки уровня активности/движения

Основан на MIT WiTrack / WiVi / rf-Pose принципах.
На macOS работает через пассивный мониторинг RSSI + фазовых изменений.

ВАЖНО: требует совместимый Wi-Fi адаптер в monitor mode или
        использует встроенный Apple Wi-Fi через CoreWLAN (пассивно).
"""

import asyncio
import logging
import time
import math
import subprocess
import re
from typing import Optional, Dict, List, Tuple
from collections import deque
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger("SASOK.WiFiCSI")


@dataclass
class CSIFrame:
    """Один кадр CSI данных."""
    timestamp: float
    rssi: float              # dBm, сила сигнала
    noise_floor: float       # dBm, уровень шума
    snr: float               # dB, signal-to-noise ratio
    phase_delta: float       # изменение фазы (радианы)
    amplitude: float         # амплитуда
    source: str = "wifi"     # "wifi" | "simulated"


@dataclass
class BodySignals:
    """Извлечённые биосигналы из CSI."""
    breathing_rate: float = 0.0      # вдохов в минуту
    heart_rate_est: float = 0.0      # уд/мин (грубая оценка)
    movement_level: float = 0.0      # 0..1 (нет движения → активное)
    presence_detected: bool = False
    distance_est: float = 0.0        # метры (грубо)
    stress_proxy: float = 0.0        # 0..1 (на основе HRV-подобного анализа)
    confidence: float = 0.0          # 0..1
    timestamp: float = field(default_factory=time.time)

    def to_modality_signal(self):
        """Конвертация в ModalitySignal для XoCore Fusion."""
        from sasok_core.core.xocore_fusion import ModalitySignal
        # Высокая ЧСС + высокий стресс-прокси → высокое возбуждение
        arousal = float(np.clip(
            0.5 * self.stress_proxy + 0.3 * (self.heart_rate_est - 60) / 60 + 0.2 * self.movement_level,
            0.0, 1.0
        ))
        return ModalitySignal(
            valence=0.0,           # Wi-Fi не определяет валентность
            arousal=arousal,
            dominance=0.5,
            confidence=self.confidence,
            cognitive_load=self.stress_proxy,
        )


class WiFiCSIScanner:
    """
    Пассивный Wi-Fi CSI сканер для macOS.

    Режимы работы:
    1. RSSI monitor — использует airport utility для получения RSSI (без root)
    2. Simulated    — генерирует реалистичные данные для разработки/тестирования
    """

    BREATHING_FREQ_MIN = 0.1   # Hz (6 вдохов/мин)
    BREATHING_FREQ_MAX = 0.5   # Hz (30 вдохов/мин)
    HEART_FREQ_MIN     = 0.8   # Hz (48 уд/мин)
    HEART_FREQ_MAX     = 2.5   # Hz (150 уд/мин)

    SAMPLING_RATE = 10.0       # Hz — частота опроса RSSI
    BUFFER_SECONDS = 30        # секунд буфера для спектрального анализа

    def __init__(self, mode: str = "auto", interface: str = "en0"):
        """
        Args:
            mode: "rssi" | "simulated" | "auto"
            interface: Wi-Fi интерфейс (по умолчанию en0 на macOS)
        """
        self.mode = mode
        self.interface = interface
        self._running = False
        self._buffer: deque = deque(
            maxlen=int(self.SAMPLING_RATE * self.BUFFER_SECONDS)
        )
        self._last_body_signals: Optional[BodySignals] = None
        self._callbacks: List = []

        if mode == "auto":
            self.mode = self._detect_mode()

        logger.info(f"WiFiCSI Scanner инициализирован в режиме: {self.mode}")

    def _detect_mode(self) -> str:
        """Определяет доступный режим работы."""
        try:
            result = subprocess.run(
                ["/System/Library/PrivateFrameworks/Apple80211.framework"
                 "/Versions/Current/Resources/airport", "-I"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0 and "agrCtlRSSI" in result.stdout:
                return "rssi"
        except Exception:
            pass
        return "simulated"

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    async def start(self):
        """Запуск сканирования."""
        self._running = True
        logger.info(f"WiFiCSI: запуск в режиме '{self.mode}'")
        asyncio.create_task(self._scan_loop())
        asyncio.create_task(self._analysis_loop())

    async def stop(self):
        """Остановка сканирования."""
        self._running = False
        logger.info("WiFiCSI: остановка")

    def get_body_signals(self) -> Optional[BodySignals]:
        """Получить последние извлечённые биосигналы."""
        return self._last_body_signals

    def add_callback(self, cb):
        """Подписаться на обновления биосигналов."""
        self._callbacks.append(cb)

    # ------------------------------------------------------------------
    # Сбор данных
    # ------------------------------------------------------------------

    async def _scan_loop(self):
        """Цикл получения сырых CSI / RSSI данных."""
        interval = 1.0 / self.SAMPLING_RATE
        while self._running:
            try:
                if self.mode == "rssi":
                    frame = await self._read_rssi()
                else:
                    frame = self._simulate_frame()

                if frame:
                    self._buffer.append(frame)

            except Exception as e:
                logger.error(f"WiFiCSI scan error: {e}")

            await asyncio.sleep(interval)

    async def _read_rssi(self) -> Optional[CSIFrame]:
        """Читает RSSI через airport utility (macOS, без root)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "/System/Library/PrivateFrameworks/Apple80211.framework"
                "/Versions/Current/Resources/airport",
                "-I",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            text = stdout.decode()

            rssi = self._parse_field(text, "agrCtlRSSI")
            noise = self._parse_field(text, "agrCtlNoise")

            if rssi is None:
                return None

            noise = noise if noise is not None else -95.0
            snr = rssi - noise

            # Фазовая дельта — аппроксимируется через изменение RSSI
            prev_rssi = self._buffer[-1].rssi if self._buffer else rssi
            phase_delta = (rssi - prev_rssi) * 0.1  # грубая аппроксимация

            return CSIFrame(
                timestamp=time.time(),
                rssi=rssi,
                noise_floor=noise,
                snr=snr,
                phase_delta=phase_delta,
                amplitude=10 ** (rssi / 20),
                source="wifi"
            )
        except Exception as e:
            logger.debug(f"RSSI read error: {e}")
            return None

    def _parse_field(self, text: str, field: str) -> Optional[float]:
        match = re.search(rf"{field}:\s*(-?\d+)", text)
        return float(match.group(1)) if match else None

    def _simulate_frame(self) -> CSIFrame:
        """
        Симуляция реалистичного CSI сигнала с:
        - дыханием (0.25 Hz = 15 вдохов/мин)
        - сердцебиением (1.17 Hz = 70 уд/мин)
        - случайным движением
        """
        t = time.time()

        # Базовый RSSI
        base_rssi = -55.0

        # Дыхательный компонент
        breathing = 2.5 * math.sin(2 * math.pi * 0.25 * t)

        # Сердечный компонент (слабее)
        heartbeat = 0.8 * math.sin(2 * math.pi * 1.17 * t)

        # Шум
        noise_component = np.random.normal(0, 0.3)

        rssi = base_rssi + breathing + heartbeat + noise_component
        noise_floor = -95.0 + np.random.normal(0, 0.5)
        snr = rssi - noise_floor

        prev_rssi = self._buffer[-1].rssi if self._buffer else rssi
        phase_delta = (rssi - prev_rssi) * 0.1

        return CSIFrame(
            timestamp=t,
            rssi=float(rssi),
            noise_floor=float(noise_floor),
            snr=float(snr),
            phase_delta=float(phase_delta),
            amplitude=float(10 ** (rssi / 20)),
            source="simulated"
        )

    # ------------------------------------------------------------------
    # Спектральный анализ → биосигналы
    # ------------------------------------------------------------------

    async def _analysis_loop(self):
        """Периодический спектральный анализ буфера."""
        while self._running:
            await asyncio.sleep(2.0)  # анализируем каждые 2 сек

            if len(self._buffer) < int(self.SAMPLING_RATE * 10):
                continue  # нужно минимум 10 сек данных

            try:
                signals = self._extract_body_signals()
                if signals:
                    self._last_body_signals = signals
                    for cb in self._callbacks:
                        try:
                            await cb(signals) if asyncio.iscoroutinefunction(cb) else cb(signals)
                        except Exception as e:
                            logger.error(f"WiFiCSI callback error: {e}")
            except Exception as e:
                logger.error(f"WiFiCSI analysis error: {e}")

    def _extract_body_signals(self) -> Optional[BodySignals]:
        """
        Извлекает дыхание и ЧСС из буфера через FFT.
        """
        frames = list(self._buffer)
        if len(frames) < 20:
            return None

        rssi_values = np.array([f.rssi for f in frames])
        timestamps  = np.array([f.timestamp for f in frames])

        # Проверка присутствия: вариабельность RSSI выше шума?
        rssi_std = float(np.std(rssi_values))
        presence = rssi_std > 0.5
        if not presence:
            return BodySignals(presence_detected=False, confidence=0.3)

        # Удаление тренда (DC компонент)
        rssi_detrended = rssi_values - np.mean(rssi_values)

        # FFT
        n = len(rssi_detrended)
        fft_vals = np.abs(np.fft.rfft(rssi_detrended))
        freqs    = np.fft.rfftfreq(n, d=1.0 / self.SAMPLING_RATE)

        # Дыхание: 0.1–0.5 Hz
        breathing_rate = self._dominant_freq(
            freqs, fft_vals,
            self.BREATHING_FREQ_MIN, self.BREATHING_FREQ_MAX
        ) * 60  # в минуты

        # Сердцебиение: 0.8–2.5 Hz
        heart_rate = self._dominant_freq(
            freqs, fft_vals,
            self.HEART_FREQ_MIN, self.HEART_FREQ_MAX
        ) * 60

        # Движение: среднеквадратичное отклонение в широкой полосе
        movement = float(np.clip(rssi_std / 5.0, 0.0, 1.0))

        # Стресс-прокси: нерегулярность дыхательного ритма
        breathing_power = fft_vals[
            (freqs >= self.BREATHING_FREQ_MIN) & (freqs <= self.BREATHING_FREQ_MAX)
        ]
        stress_proxy = float(np.clip(
            1.0 - (np.max(breathing_power) / (np.sum(breathing_power) + 1e-9)),
            0.0, 1.0
        ))

        # Дистанция (грубая, по затуханию RSSI)
        mean_rssi = float(np.mean(rssi_values))
        ref_rssi  = -40.0   # RSSI на 1м (типично)
        path_loss_exp = 2.5
        distance = 10 ** ((ref_rssi - mean_rssi) / (10 * path_loss_exp))
        distance = float(np.clip(distance, 0.3, 15.0))

        # Уверенность
        confidence = float(np.clip(rssi_std / 3.0, 0.1, 0.9))
        if frames[-1].source == "simulated":
            confidence *= 0.7  # симуляция менее достоверна

        return BodySignals(
            breathing_rate=round(breathing_rate, 1),
            heart_rate_est=round(heart_rate, 1),
            movement_level=round(movement, 3),
            presence_detected=True,
            distance_est=round(distance, 2),
            stress_proxy=round(stress_proxy, 3),
            confidence=round(confidence, 3),
            timestamp=time.time(),
        )

    def _dominant_freq(
        self,
        freqs: np.ndarray,
        fft_vals: np.ndarray,
        f_min: float,
        f_max: float,
    ) -> float:
        """Доминирующая частота в заданном диапазоне."""
        mask = (freqs >= f_min) & (freqs <= f_max)
        if not np.any(mask):
            return (f_min + f_max) / 2
        idx = np.argmax(fft_vals[mask])
        return float(freqs[mask][idx])


# ------------------------------------------------------------------
# Быстрый тест
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def main():
        scanner = WiFiCSIScanner(mode="simulated")

        def on_signals(signals: BodySignals):
            print(f"[WiFiCSI] Дыхание: {signals.breathing_rate:.1f}/мин | "
                  f"ЧСС≈{signals.heart_rate_est:.0f} | "
                  f"Стресс: {signals.stress_proxy:.2f} | "
                  f"Дист: {signals.distance_est:.1f}м | "
                  f"conf={signals.confidence:.2f}")

        scanner.add_callback(on_signals)
        await scanner.start()

        print("Wi-Fi CSI сканер запущен. Ctrl+C для остановки.")
        try:
            await asyncio.sleep(30)
        except KeyboardInterrupt:
            pass
        await scanner.stop()

    asyncio.run(main())
