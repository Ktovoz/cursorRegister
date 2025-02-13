import sys
import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional, Callable, List, Tuple
from dataclasses import dataclass
from cursor_account_generator import generate_cursor_account
from cursor_id_resetter import CursorResetter
from cursor_auth_updater import CursorAuthUpdater
from loguru import logger
from dotenv import load_dotenv
import os
from pathlib import Path
from cursor_utils import (
    PathManager, FileManager, Result, UIManager, 
    StyleManager, MessageManager, error_handler, EnvManager
)
import functools

@dataclass
class WindowConfig:
    width: int = 480
    height: int = 390
    title: str = "Cursor账号管理工具"
    backup_dir: str = "env_backups"
    max_backups: int = 10
    env_vars: List[Tuple[str, str]] = None
    buttons: List[Tuple[str, str]] = None

    def __post_init__(self):
        self.env_vars = [
            ('DOMAIN', '域名'), 
            ('EMAIL', '邮箱'), 
            ('PASSWORD', '密码')
        ]
        self.buttons = [
            ("生成账号", "generate_account"),
            ("重置ID", "reset_ID"),
            ("更新账号信息", "update_auth")
        ]

class CursorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.window_config = WindowConfig()
        self.root.title(self.window_config.title)
        self._configure_window()
        self._init_variables()
        self.setup_ui()

    def _configure_window(self) -> None:
        UIManager.center_window(self.root, self.window_config.width, self.window_config.height)
        self.root.resizable(False, False)
        self.root.configure(bg='#FFFFFF')
        if os.name == 'nt':
            self.root.attributes('-alpha', 0.98)
        
    def _init_variables(self) -> None:
        self.entries: Dict[str, ttk.Entry] = {}
        self.main_frame: Optional[ttk.Frame] = None
        StyleManager.setup_styles()

    def setup_ui(self) -> None:
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self._create_frames()

    def _create_frames(self) -> None:
        account_frame = UIManager.create_labeled_frame(self.main_frame, "账号信息")
        for row, (var_name, label_text) in enumerate(self.window_config.env_vars):
            entry = UIManager.create_labeled_entry(account_frame, label_text, row)
            entry.bind('<FocusIn>', functools.partial(self._handle_focus_in, entry))
            entry.bind('<FocusOut>', functools.partial(self._handle_focus_out, entry))
            if os.getenv(var_name):
                entry.insert(0, os.getenv(var_name))
            self.entries[var_name] = entry

        cookie_frame = UIManager.create_labeled_frame(self.main_frame, "Cookie设置")
        self.entries['cookie'] = UIManager.create_labeled_entry(cookie_frame, "Cookie", 0)
        self.entries['cookie'].insert(0, "WorkosCursorSessionToken")

        self._create_button_frame()

    def _create_button_frame(self) -> None:
        button_frame = ttk.Frame(self.main_frame, style='TFrame')
        button_frame.pack(fill=tk.X, pady=(15,0))

        container = ttk.Frame(button_frame, style='TFrame')
        container.pack()

        for col, (text, command) in enumerate(self.window_config.buttons):
            btn = ttk.Button(
                container,
                text=text,
                command=getattr(self, command),
                style='Custom.TButton'
            )
            btn.grid(row=0, column=col, padx=10)

        footer_frame = ttk.Frame(button_frame, style='TFrame')
        footer_frame.pack(fill=tk.X, pady=(10,5))
        ttk.Label(
            footer_frame,
            text="powered by kto 仅供学习使用",
            style='Footer.TLabel'
        ).pack()

    @error_handler
    def generate_account(self) -> None:
        self.backup_env_file()
        
        updates = {}
        if domain := self.entries['DOMAIN'].get().strip():
            updates['DOMAIN'] = domain
            result = EnvManager.update_env_vars(updates)
            if not result:
                raise RuntimeError(f"保存域名失败: {result.message}")
            load_dotenv(override=True)

        result = generate_cursor_account()
        if isinstance(result, Result):
            if result:
                email, password = result.data
                self._update_entry_values(email, password)
                MessageManager.show_success(self.root, "账号生成成功")
                self._save_env_vars()
            else:
                raise RuntimeError(result.message)
        else:
            email, password = result
            self._update_entry_values(email, password)
            MessageManager.show_success(self.root, "账号生成成功")
            self._save_env_vars()

    def _save_env_vars(self) -> None:
        updates = {}
        for var_name, _ in self.window_config.env_vars:
            if value := self.entries[var_name].get().strip():
                updates[var_name] = value
        
        if updates:
            result = EnvManager.update_env_vars(updates)
            if not result:
                MessageManager.show_warning(self.root, f"保存环境变量失败: {result.message}")

    @error_handler
    def reset_ID(self) -> None:
        resetter = CursorResetter()
        result = resetter.reset()
        if result:
            MessageManager.show_success(self.root, result.message)
            self._save_env_vars()
        else:
            raise Exception(result.message)

    def backup_env_file(self) -> None:
        env_path = PathManager.get_env_path()
        if not env_path.exists():
            raise Exception(f"未找到.env文件: {env_path}")

        backup_dir = Path(self.window_config.backup_dir)
        result = FileManager.backup_file(env_path, backup_dir, '.env', self.window_config.max_backups)
        if not result:
            raise Exception(result.message)

    @error_handler
    def update_auth(self) -> None:
        cookie_str = self.entries['cookie'].get().strip()
        if not self._validate_cookie(cookie_str):
            return

        self.backup_env_file()
        updater = CursorAuthUpdater()
        result = updater.process_cookies(cookie_str)
        if result:
            MessageManager.show_success(self.root, result.message)
            self.entries['cookie'].delete(0, tk.END)
            self._save_env_vars()
        else:
            raise Exception(result.message)

    def _validate_cookie(self, cookie_str: str) -> bool:
        if not cookie_str:
            MessageManager.show_warning(self.root, "请输入Cookie字符串")
            return False

        if "WorkosCursorSessionToken=" not in cookie_str:
            MessageManager.show_warning(self.root, "Cookie字符串格式不正确，必须包含 WorkosCursorSessionToken")
            return False

        return True

    def _update_entry_values(self, email: str, password: str) -> None:
        self.entries['EMAIL'].delete(0, tk.END)
        self.entries['EMAIL'].insert(0, email)
        self.entries['PASSWORD'].delete(0, tk.END)
        self.entries['PASSWORD'].insert(0, password)

    def _handle_focus_in(self, entry: ttk.Entry, event) -> None:
        entry.configure(style='TEntry')

    def _handle_focus_out(self, entry: ttk.Entry, event) -> None:
        if not entry.get().strip():
            entry.configure(style='TEntry')

def setup_logging() -> None:
    logger.remove()
    logger.add(
        sink=Path("./cursorRegister_log") / "{time:YYYY-MM-DD_HH}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} |{level:8}| - {message}",
        rotation="10 MB",
        retention="14 days",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        level="DEBUG"
    )
    logger.add(
        sink=sys.stderr,
        colorize=True,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        level="DEBUG"
    )

def main() -> None:
    try:
        env_path = PathManager.get_env_path()
        load_dotenv(dotenv_path=env_path)
        setup_logging()
        root = tk.Tk()
        app = CursorApp(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"程序启动失败: {e}")
        MessageManager.show_error(root, "程序启动失败", e)

if __name__ == "__main__":
    main()
