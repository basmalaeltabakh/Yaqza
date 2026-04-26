from src.serving.app import create_app


def test_create_app_returns_object():
    app = create_app()
    assert hasattr(app, "name")
    assert app.name == "yaqza-serving"
