import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
APP = Path(__file__).parent / "app.py"


def test_app_has_text_area():
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert len(app.get("text_area")) == 1
    assert app.get("text_area")[0].label == "Nhập câu tiếng Pháp"


def test_translate_fake_backend():
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["demo_fake_mode"] = True
    app.run(timeout=30)
    app.text_area[0].set_value("Bonjour le monde")
    app.button[0].click()
    app.run(timeout=30)
    assert not app.exception
    assert any("Hello world" in item.value for item in app.markdown)


def test_dictionary_toggle_and_translate():
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.session_state["demo_fake_mode"] = True
    app.toggle[0].set_value(True)
    app.text_area[0].set_value("Bonjour le monde")
    app.button[0].click()
    app.run(timeout=30)
    assert app.toggle[0].value is True


def test_clear_button():
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.session_state["demo_fake_mode"] = True
    app.text_area[0].set_value("Bonjour")
    app.run(timeout=30)
    app.button[1].click()
    app.run(timeout=30)
    assert app.text_area[0].value == ""
