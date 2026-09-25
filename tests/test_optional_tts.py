import unittest

from src import voz


class TestOptionalTTS(unittest.TestCase):
    def test_tts_requires_optional_dependency(self):
        with self.assertRaisesRegex(RuntimeError, "tensorflow-tts|soundfile"):
            voz.TextToSpeechSynthesizer()


if __name__ == "__main__":
    unittest.main()
