import json
import os

class Config:
    def __init__(self):
        self.config_file = 'config.json'
        self.default_config = {
            "gaze_neutral_x": 0.0,
            "gaze_neutral_y": 0.0,
            "gaze_deadzone": 0.05,
            "iris_gain": 3.0,
            "iris_gain_y": 1.0,
            "smoothing_alpha": 0.3,
            "hand_gain": 2.0,
            "fist_threshold": 0.1,
            "fist_release": 0.2,
            "super_charge_move_min": 0.02,
            "super_charge_rate": 2.0,
            "super_blink_count": 3,
            "super_blink_window": 2.0,
            "smile_hysteresis": 0.8,
            "level_duration": 60,
            "best_score": 0
        }
        self.config = self.load_config()
        # Установка атрибутов для совместимости
        self._update_attrs()

    def _update_attrs(self):
        """Обновляет атрибуты класса для совместимости с кодом"""
        for key, value in self.config.items():
            setattr(self, key, value)

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                try:
                    return json.load(f)
                except:
                    return self.default_config
        else:
            self.save_config(self.default_config)
            return self.default_config

    def save_config(self, config):
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config(self.config)
        # Обновляем атрибут
        setattr(self, key, value)

    def update(self, **kwargs):
        self.config.update(kwargs)
        self.save_config(self.config)
        # Обновляем атрибуты
        for key, value in kwargs.items():
            setattr(self, key, value)

    def as_dict(self):
        return self.config

# Синглтон для конфигурации
CFG = Config()