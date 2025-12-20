from PyQt6.QtWidgets import QSystemTrayIcon


class Notifier:
    def __init__(self, tray: QSystemTrayIcon):
        self.tray = tray

    def info(self, title: str, msg: str, ms: int = 5000) -> None:
        self.tray.showMessage(title, msg, QSystemTrayIcon.MessageIcon.Information, ms)

    def warn(self, title: str, msg: str, ms: int = 8000) -> None:
        self.tray.showMessage(title, msg, QSystemTrayIcon.MessageIcon.Warning, ms)
