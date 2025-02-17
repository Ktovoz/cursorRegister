import os
import tkinter as tk
from tkinter import ttk, messagebox
from loguru import logger
from datetime import datetime
from typing import Dict, List, Tuple, Callable
import glob
import csv
import threading
from registerAc import CursorRegistration
from utils import CursorManager, error_handler
from .ui import UI


class ManageTab(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, style='TFrame', **kwargs)
        self.registrar = None
        self.setup_ui()

    def setup_ui(self):
        accounts_frame = UI.create_labeled_frame(self, "已保存账号")
        
        columns = ('域名', '邮箱', '额度', '剩余天数')
        tree = ttk.Treeview(accounts_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        tree.bind('<<TreeviewSelect>>', self.on_select)
        
        scrollbar = ttk.Scrollbar(accounts_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = ttk.Frame(self, style='TFrame')
        button_frame.pack(pady=(8, 0), padx=2)

        container_frame = ttk.Frame(button_frame, style='TFrame')
        container_frame.pack(expand=True)

 
        first_button_frame = ttk.Frame(container_frame, style='TFrame')
        first_button_frame.pack(fill=tk.X, expand=True, pady=(0, 5))
        
        second_button_frame = ttk.Frame(container_frame, style='TFrame')
        second_button_frame.pack(fill=tk.X, expand=True, pady=(5, 5))

        third_button_frame = ttk.Frame(container_frame, style='TFrame')
        third_button_frame.pack(fill=tk.X, expand=True, pady=(5, 10))

        buttons = [
            ("刷新列表", self.refresh_list),
            ("获取试用信息", self.show_trial_info),
            ("刷新cookie", self.update_auth),
            ("重置机器ID", self.reset_machine_id),
            ("删除账号", self.delete_account)
        ]


        # placeholder_buttons = [
        #     ("占位按钮1", None),
        #     ("占位按钮2", None),
        #     ("占位按钮3", None),
        #     ("占位按钮4", None)
        # ]
        # buttons.extend(placeholder_buttons)

        frames = [first_button_frame, second_button_frame, third_button_frame]
        for i, (text, command) in enumerate(buttons):
            frame_index = i // 3  
            if frame_index >= len(frames):
                break
            
            btn = ttk.Button(
                frames[frame_index],
                text=text,
                command=command,
                style='Custom.TButton',
                width=15,
                state='normal' if command else 'disabled'
            )
            btn.pack(side=tk.LEFT, padx=3, expand=True)

        self.account_tree = tree
        self.selected_item = None

    def on_select(self, event):
        selected_items = self.account_tree.selection()
        if selected_items:
            self.selected_item = selected_items[0]
        else:
            self.selected_item = None

    def get_csv_files(self) -> List[str]:
        try:
            return glob.glob('env_backups/cursor_account_*.csv')
        except Exception as e:
            logger.error(f"获取CSV文件列表失败: {str(e)}")
            return []

    def parse_csv_file(self, csv_file: str) -> Dict[str, str]:
        account_data = {
            'DOMAIN': '', 
            'EMAIL': '', 
            'COOKIES_STR': '', 
            'QUOTA': '未知',
            'DAYS': '未知',
            'PASSWORD': '',
            'API_KEY': '',
            'MOE_MAIL_URL': ''
        }
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)
                for row in csv_reader:
                    if len(row) >= 2:
                        key, value = row[0], row[1]
                        if key in account_data:
                            account_data[key] = value
        except Exception as e:
            logger.error(f"解析文件 {csv_file} 失败: {str(e)}")
        return account_data

    def update_csv_file(self, csv_file: str, quota: str, days: str) -> None:
        try:
            rows = []
            with open(csv_file, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
                rows = list(csv_reader)

            quota_found = days_found = False
            for row in rows:
                if len(row) >= 2:
                    if row[0] == 'QUOTA':
                        row[1] = quota
                        quota_found = True
                    elif row[0] == 'DAYS':
                        row[1] = days
                        days_found = True

            if not quota_found:
                rows.append(['QUOTA', quota])
            if not days_found:
                rows.append(['DAYS', days])

            with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerows(rows)

            logger.debug(f"已更新CSV文件: {csv_file}")
        except Exception as e:
            logger.error(f"更新CSV文件失败: {str(e)}")
            raise

    def refresh_list(self):
        for item in self.account_tree.get_children():
            self.account_tree.delete(item)
        
        try:
            csv_files = self.get_csv_files()
            for csv_file in csv_files:
                account_data = self.parse_csv_file(csv_file)
                self.account_tree.insert('', 'end', iid=csv_file, values=(
                    account_data['DOMAIN'],
                    account_data['EMAIL'],
                    account_data['QUOTA'],
                    account_data['DAYS']
                ))
            
            logger.info("账号列表已刷新")
        except Exception as e:
            UI.show_error(self.winfo_toplevel(), "刷新列表失败", e)

    def get_selected_account(self) -> Tuple[str, Dict[str, str]]:
        if not self.selected_item:
            raise ValueError("请先选择要操作的账号")

        item_values = self.account_tree.item(self.selected_item)['values']
        if not item_values or len(item_values) < 2:
            raise ValueError("所选账号信息不完整")

        csv_file_path = self.selected_item
        account_data = self.parse_csv_file(csv_file_path)
        
        if not account_data['DOMAIN'] or not account_data['EMAIL']:
            raise ValueError("账号信息不完整")
            
        return csv_file_path, account_data

    def handle_account_action(self, action_name: str, action: Callable[[str, Dict[str, str]], None]) -> None:
        try:
            csv_file_path, account_data = self.get_selected_account()
            action(csv_file_path, account_data)
        except Exception as e:
            UI.show_error(self.winfo_toplevel(), f"{action_name}失败", e)
            logger.error(f"{action_name}失败: {str(e)}")

    def show_trial_info(self):
        def get_trial_info(csv_file_path: str, account_data: Dict[str, str]) -> None:
            cookie_str = account_data.get('COOKIES_STR', '')
            if not cookie_str:
                raise ValueError(f"未找到账号 {account_data['EMAIL']} 的Cookie信息")

            def fetch_and_display_info():
                try:
                    logger.debug("开始获取试用信息...")
                    logger.debug(f"获取到的cookie字符串长度: {len(cookie_str) if cookie_str else 0}")

                    if "WorkosCursorSessionToken=" not in cookie_str:
                        logger.debug("Cookie字符串中未包含WorkosCursorSessionToken前缀，正在添加...")
                        cookie_str_with_prefix = f"WorkosCursorSessionToken={cookie_str}"
                    else:
                        cookie_str_with_prefix = cookie_str
                    
                    logger.debug("正在初始化浏览器...")
                    self.registrar = CursorRegistration()
                    self.registrar.headless = True
                    self.registrar.init_browser()
                    logger.debug("浏览器初始化完成")
                    
                    logger.debug("正在获取试用信息...")
                    trial_info = self.registrar.get_trial_info(cookie=cookie_str_with_prefix)
                    quota, days = trial_info[0], trial_info[1]
                    logger.info(f"成功获取试用信息: 额度={quota}, 天数={days}")
                    
                    self.account_tree.set(self.selected_item, '额度', quota)
                    self.account_tree.set(self.selected_item, '剩余天数', f"{days}")
                    
                    try:
                        rows = []
                        with open(csv_file_path, 'r', encoding='utf-8') as f:
                            csv_reader = csv.reader(f)
                            rows = list(csv_reader)

                        quota_found = days_found = False
                        for row in rows:
                            if len(row) >= 2:
                                if row[0] == 'QUOTA':
                                    row[1] = str(quota)
                                    quota_found = True
                                elif row[0] == 'DAYS':
                                    row[1] = str(days)
                                    days_found = True

                        if not quota_found:
                            rows.append(['QUOTA', str(quota)])
                        if not days_found:
                            rows.append(['DAYS', str(days)])

                        with open(csv_file_path, 'w', encoding='utf-8', newline='') as f:
                            csv_writer = csv.writer(f)
                            csv_writer.writerows(rows)
                        
                        logger.debug(f"已更新CSV文件: {csv_file_path}")
                    except Exception as e:
                        logger.error(f"更新CSV文件失败: {str(e)}")
                        raise ValueError(f"更新CSV文件失败: {str(e)}")
                    
                    self.registrar.browser.quit()
                    logger.debug("浏览器已关闭")
                    
                    self.winfo_toplevel().after(0, lambda: UI.show_success(
                        self.winfo_toplevel(),
                        f"账户可用额度: {quota}\n剩余天数: {days}"
                    ))
                except Exception as e:
                    error_message = str(e)
                    logger.error(f"获取试用信息失败: {error_message}")
                    logger.exception("详细错误信息:")
                    self.winfo_toplevel().after(0, lambda: UI.show_error(
                        self.winfo_toplevel(), 
                        "获取账号信息失败", 
                        error_message
                    ))
                finally:
                    if hasattr(self, 'registrar') and self.registrar and self.registrar.browser:
                        self.registrar.browser.quit()

            logger.debug("开始获取信息...")
            threading.Thread(target=fetch_and_display_info, daemon=True).start()

        self.handle_account_action("获取试用信息", get_trial_info)

    def update_auth(self) -> None:
        def update_account_auth(csv_file_path: str, account_data: Dict[str, str]) -> None:
            cookie_str = account_data.get('COOKIES_STR', '')
            email = account_data.get('EMAIL', '')
            if not cookie_str:
                raise ValueError(f"未找到账号 {email} 的Cookie信息")

            if "WorkosCursorSessionToken=" not in cookie_str:
                cookie_str = f"WorkosCursorSessionToken={cookie_str}"

            result = CursorManager().process_cookies(cookie_str, email)
            if not result.success:
                raise ValueError(result.message)

            UI.show_success(self.winfo_toplevel(), f"账号 {email} 的Cookie已刷新")
            logger.info(f"已刷新账号 {email} 的Cookie")

        self.handle_account_action("刷新Cookie", update_account_auth)

    def delete_account(self):
        def delete_account_file(csv_file_path: str, account_data: Dict[str, str]) -> None:
            confirm_message = (
                f"确定要删除以下账号吗？\n\n"
                f"域名：{account_data['DOMAIN']}\n"
                f"邮箱：{account_data['EMAIL']}\n"
                f"额度：{account_data['QUOTA']}\n"
                f"剩余天数：{account_data['DAYS']}"
            )
            
            if not messagebox.askyesno("确认删除", confirm_message, icon='warning'):
                return

            try:
                os.remove(csv_file_path)
                self.account_tree.delete(self.selected_item)
                logger.info(f"已删除账号: {account_data['DOMAIN']} - {account_data['EMAIL']}")
                UI.show_success(self.winfo_toplevel(), 
                              f"已删除账号: {account_data['DOMAIN']} - {account_data['EMAIL']}")
            except Exception as e:
                raise Exception(f"删除文件失败: {str(e)}")

        self.handle_account_action("删除账号", delete_account_file)

    @error_handler
    def reset_machine_id(self) -> None:
        try:
            result = CursorManager.reset()
            if not result:
                raise Exception(result.message)
            UI.show_success(self.winfo_toplevel(), result.message)
        except Exception as e:
            UI.show_error(self.winfo_toplevel(), "重置机器ID失败", e) 