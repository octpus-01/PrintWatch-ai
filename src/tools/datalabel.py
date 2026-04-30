import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk

class ImageLabelTool:
    def __init__(self, root):
        self.root = root
        self.root.title("图片快速标注工具 - 带预览版")
        self.root.geometry("900x650")
        self.root.resizable(False, False)

        # ========== 核心变量 ==========
        self.file_dir = ""       # 文件目录
        self.file_list = []      # 图片文件列表
        self.current_idx = 0     # 当前索引
        self.label_list = ["正常", "异常", "可疑", "背景", "目标"]  # 默认标签
        self.photo = None        # 用于显示图片（防止被GC回收）
        
        # 支持的图片格式（自动过滤非图片）
        self.IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

        # ========== 界面布局 ==========
        self.setup_ui()
        self.bind_shortcuts()  # 绑定键盘快捷键

    def setup_ui(self):
        # 顶部：目录选择
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="图片目录：").grid(row=0, column=0, padx=5)
        self.dir_entry = ttk.Entry(top_frame, width=45)
        self.dir_entry.grid(row=0, column=1, padx=5)
        ttk.Button(top_frame, text="选择文件夹", command=self.select_dir).grid(row=0, column=2, padx=5)
        ttk.Button(top_frame, text="加载图片", command=self.load_files).grid(row=0, column=3, padx=5)

        # 中间：图片预览区域
        preview_frame = ttk.Frame(self.root, padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(preview_frame, text="图片预览", font=("", 14, "bold")).pack()
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(pady=5)

        # 信息显示
        info_frame = ttk.Frame(self.root, padding=5)
        info_frame.pack(fill=tk.X)
        ttk.Label(info_frame, text="当前文件：", font=("", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.current_file_label = ttk.Label(info_frame, text="未加载图片", font=("", 11))
        self.current_file_label.grid(row=0, column=1, padx=5, sticky="w")

        ttk.Label(info_frame, text="进度：", font=("", 12, "bold")).grid(row=1, column=0, sticky="w")
        self.progress_label = ttk.Label(info_frame, text="0/0", font=("", 11))
        self.progress_label.grid(row=1, column=1, padx=5, sticky="w")

        # 标签配置区
        label_frame = ttk.Frame(self.root, padding=5)
        label_frame.pack(fill=tk.X)
        ttk.Label(label_frame, text="自定义标签（空格分隔）：").pack(anchor=tk.W)
        self.label_entry = ttk.Entry(label_frame, width=60)
        self.label_entry.insert(0, " ".join(self.label_list))
        self.label_entry.pack(pady=3, anchor=tk.W)
        ttk.Button(label_frame, text="更新标签", command=self.update_labels).pack(anchor=tk.W)

        # 快捷键标签按钮区
        ttk.Label(self.root, text="【快捷键标注】", font=("", 12, "bold")).pack(pady=5)
        self.shortcut_frame = ttk.Frame(self.root)
        self.shortcut_frame.pack()

        # 底部：控制按钮
        bottom_frame = ttk.Frame(self.root, padding=10)
        bottom_frame.pack(fill=tk.X)
        ttk.Button(bottom_frame, text="上一个 (←)", command=self.prev_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="下一个 (→)", command=self.next_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="退出", command=self.root.quit).pack(side=tk.RIGHT, padx=5)

    def bind_shortcuts(self):
        """绑定键盘快捷键"""
        self.root.bind("<Left>", lambda e: self.prev_file())   # 左箭头：上一个
        self.root.bind("<Right>", lambda e: self.next_file()) # 右箭头：下一个
        self.root.bind("<Key>", self.key_label)               # 数字键：快速标注

    def select_dir(self):
        """选择标注文件夹"""
        directory = filedialog.askdirectory()
        if directory:
            self.file_dir = directory
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, directory)

    def load_files(self):
        """仅加载图片文件，自动过滤其他文件"""
        if not self.file_dir:
            messagebox.showwarning("提示", "请先选择图片文件夹！")
            return

        # 只保留图片文件
        self.file_list = [
            f for f in os.listdir(self.file_dir)
            if os.path.isfile(os.path.join(self.file_dir, f))
            and f.lower().endswith(self.IMG_EXTENSIONS)
        ]
        self.file_list = sorted(self.file_list)
        self.current_idx = 0

        if not self.file_list:
            messagebox.showinfo("提示", "文件夹内无支持的图片！")
            return

        self.show_current_file()
        messagebox.showinfo("成功", f"加载完成，共 {len(self.file_list)} 张图片")

    def show_current_file(self):
        """显示当前图片和信息"""
        if not self.file_list:
            return
            
        filename = self.file_list[self.current_idx]
        self.current_file_label.config(text=filename)
        self.progress_label.config(text=f"{self.current_idx + 1}/{len(self.file_list)}")
        
        # 显示图片预览
        self.show_image_preview()
        self.refresh_label_buttons()

    def show_image_preview(self):
        """自适应显示图片预览"""
        img_path = os.path.join(self.file_dir, self.file_list[self.current_idx])
        
        try:
            # 打开图片并等比例缩放（最大600x400）
            img = Image.open(img_path)
            img.thumbnail((600, 400))
            self.photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self.photo)
        except Exception as e:
            self.preview_label.config(text="图片预览失败", image="")

    def refresh_label_buttons(self):
        """刷新标签按钮（修复此处）"""
        # 错误写法：self.shortcut_frame.wchildren()
        # 正确写法：
        for widget in self.shortcut_frame.winfo_children():
            widget.destroy()

        for i, label in enumerate(self.label_list):
            btn = ttk.Button(
                self.shortcut_frame,
                text=f"{i+1} - {label}",
                command=lambda l=label: self.do_label(l)
            )
            btn.grid(row=i//3, column=i%3, padx=5, pady=3)

    def update_labels(self):
        """更新自定义标签"""
        text = self.label_entry.get().strip()
        if not text:
            messagebox.showwarning("提示", "标签不能为空")
            return
        self.label_list = text.split()
        self.refresh_label_buttons()

    def key_label(self, event):
        """数字键快速标注"""
        if event.char.isdigit():
            num = int(event.char) - 1
            if 0 <= num < len(self.label_list):
                self.do_label(self.label_list[num])

    def do_label(self, label):
        """执行标注：重命名图片"""
        if not self.file_list:
            return

        old_name = self.file_list[self.current_idx]
        old_path = os.path.join(self.file_dir, old_name)

        stem = Path(old_name).stem
        suffix = Path(old_name).suffix
        new_name = f"{stem}_{label}{suffix}"
        new_path = os.path.join(self.file_dir, new_name)

        # 重名处理
        if os.path.exists(new_path):
            new_name = f"{stem}_{label}_{self.current_idx}{suffix}"
            new_path = os.path.join(self.file_dir, new_name)

        try:
            os.rename(old_path, new_path)
            self.file_list[self.current_idx] = new_name
            self.show_current_file()
            # messagebox.showinfo("标注成功", f"已标注：{label}")
            self.next_file()
        except Exception as e:
            messagebox.showerror("错误", f"标注失败：{str(e)}")

    def prev_file(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current_file()

    def next_file(self):
        if self.current_idx < len(self.file_list) - 1:
            self.current_idx += 1
            self.show_current_file()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelTool(root)
    root.mainloop()