import os
import csv
import random
import logging
import requests
import numpy as np
import pandas as pd
import tkinter as tk
import multiprocessing
from tkinter import ttk
from threading import Thread
from bs4 import BeautifulSoup
from ttkthemes import ThemedTk
import matplotlib.pyplot as plt
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://205.174.165.80/IOTDataset/CIC_IOT_Dataset2023/Dataset/CSV/CSV/"
SAVE_DIR = "downloads"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_DIR = "logs/app/"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("processing_logger")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(LOG_DIR, "processing.log"), mode="w", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)


def get_links(url):
    """Получаем все ссылки (href) со страницы."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return [a["href"] for a in soup.find_all("a", href=True)]


def download_file(file_url, file_path):
    """Скачиваем один файл по URL в file_path (если не существует)."""
    if os.path.exists(file_path):
        print(f"🔵 Файл уже существует, пропускаем: {file_path}")
        return
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Загружен: {file_path}")
    except Exception as e:
        print(f"Ошибка загрузки {file_url}: {e}")


def collect_csv_files(remote_url, local_subpath):
    """Рекурсивно собираем все .csv файлы из удалённой директории."""
    files = []
    items = get_links(remote_url)
    for item in items:
        if item.endswith('/'):
            new_remote = remote_url + item
            new_local = os.path.join(local_subpath, item.rstrip('/'))
            files.extend(collect_csv_files(new_remote, new_local))
        elif item.endswith('.csv'):
            remote_file = remote_url + item
            local_file = os.path.join(local_subpath, item)
            files.append((remote_file, local_file))
    return files


# ------------------------------------------------
# Фрейм для скачивания датасетов
# ------------------------------------------------
class DownloadFrame(ttk.Frame):
    def __init__(self, master, all_folders):
        super().__init__(master, padding=10)
        self.all_folders = all_folders
        self.folder_vars = {}
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style(self)
        style.configure("TFrame", background="#fafafa")
        style.configure("TLabel", background="#fafafa", foreground="#000", font=("Courier", 12))
        style.configure("TCheckbutton", background="#fafafa", foreground="#000", font=("Courier", 11))
        style.configure("TButton",
                        background="#fafafa",
                        foreground="#000",
                        font=("Courier", 12, "bold"))
        style.map("TButton",
                  background=[("active", "#2d2d2d")],
                  foreground=[("active", "#000")])
        style.configure("Horizontal.TProgressbar",
                        troughcolor="#fafafa",
                        bordercolor="#fafafa",
                        background="#000",
                        lightcolor="#000",
                        darkcolor="#000")
        style.configure("Vertical.TScrollbar", troughcolor="#fafafa", background="#dfdfdf")

        title_label = ttk.Label(self, text="Выберите папки для скачивания:", font=("Courier", 14, "bold"))
        title_label.pack(pady=5)

        self.status_label = ttk.Label(self, text=f"Доступно папок: {len(self.all_folders)} | Выбрано: 0")
        self.status_label.pack(pady=5)

        checkboxes_frame = ttk.Frame(self)
        checkboxes_frame.pack(fill="both", expand=True, pady=5)

        canvas_bg_color = "#fafafa"
        self.canvas = tk.Canvas(checkboxes_frame, bg=canvas_bg_color, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(checkboxes_frame, orient="vertical",
                                       command=self.canvas.yview, style="Vertical.TScrollbar")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        scrollable_frame = ttk.Frame(self.canvas, style="TFrame")
        self.canvas_window = self.canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def on_frame_configure(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        scrollable_frame.bind("<Configure>", on_frame_configure)

        for folder in self.all_folders:
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(scrollable_frame, text=folder, variable=var, command=self.update_count)
            cb.pack(anchor="w")
            self.folder_vars[folder] = var

        self.progress_bar = ttk.Progressbar(self, mode="determinate", length=400, style="Horizontal.TProgressbar")
        self.progress_bar.pack(pady=5)

        self.progress_label = ttk.Label(self, text="Прогресс: 0/0 (0%)")
        self.progress_label.pack(pady=5)

        download_btn = ttk.Button(
            self,
            text="Скачать выбранные",
            command=self.start_download,
            style="TButton"
        )
        download_btn.pack(pady=5)

    def update_count(self):
        selected_count = sum(var.get() for var in self.folder_vars.values())
        self.status_label.config(text=f"Доступно папок: {len(self.all_folders)} | Выбрано: {selected_count}")

    def start_download(self):
        selected_folders = [folder for folder, var in self.folder_vars.items() if var.get()]
        if not selected_folders:
            messagebox.showerror("Ошибка", "Выберите хотя бы одну папку для скачивания!", parent=self)
            return
        messagebox.showinfo("Загрузка",
                            f"Скачивание файлов из {len(selected_folders)} папок началось.\n"
                            "Смотрите консоль для подробностей.",
                            parent=self)
        thread = Thread(target=self.download_thread, args=(selected_folders,), daemon=True)
        thread.start()

    def download_thread(self, selected_folders):
        all_files = []
        for folder in selected_folders:
            remote_url = BASE_URL + folder
            local_subpath = os.path.join(SAVE_DIR, folder.rstrip('/'))
            files_in_folder = collect_csv_files(remote_url, local_subpath)
            all_files.extend(files_in_folder)

        total_files = len(all_files)
        if total_files == 0:
            self.after(0, lambda: messagebox.showinfo("Результат",
                                                      "Не найдено ни одного CSV-файла в выбранных папках.",
                                                      parent=self))
            return

        def update_progress(i):
            percent = int((i / total_files) * 100)
            self.progress_bar["value"] = i
            self.progress_label.config(text=f"Прогресс: {i}/{total_files} ({percent}%)")

        self.after(0, lambda: self.progress_bar.config(value=0, maximum=total_files))
        self.after(0, lambda: self.progress_label.config(text=f"Прогресс: 0/{total_files} (0%)"))

        completed_count = 0
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_file = {
                executor.submit(download_file, file_url, file_path): (file_url, file_path)
                for file_url, file_path in all_files
            }
            for future in as_completed(future_to_file):
                completed_count += 1
                self.after(0, lambda c=completed_count: update_progress(c))

        def finish_message():
            messagebox.showinfo("Готово!", "Все выбранные файлы загружены!", parent=self)
            self.status_label.config(text="Загрузка завершена!")
        self.after(0, finish_message)


# ----------------------------------------------------------------------
# ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА ОТДЕЛЬНОГО ФАЙЛА
# ----------------------------------------------------------------------
def process_file(fpath: str, remove_incomplete: bool, remove_dup: bool):
    """
    Запускается в отдельном процессе (multiprocessing).
    Построчно читает один CSV-файл:
      - remove_incomplete: пропускает строки с пустыми полями
      - remove_dup: пропускает строки, которые уже встретились в файле
    Логирует каждый пропуск.
    Возвращает кортеж: (rows, incomplete_count, duplicates_count, header)
    """
    rows = []
    seen = set()
    incomplete_count = 0
    duplicates_count = 0
    header = None

    try:
        with open(fpath, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            row_index = 0
            for row in reader:
                row_index += 1
                if row_index == 1:
                    header = row
                    continue

                if remove_incomplete and any(col.strip() == "" for col in row):
                    logger.info(f"[Некорректная строка] File='{fpath}', Line={row_index}, Data={row}")
                    incomplete_count += 1
                    continue

                row_tuple = tuple(row)
                if remove_dup and (row_tuple in seen):
                    logger.info(f"[Дубликат] File='{fpath}', Line={row_index}, Data={row}")
                    duplicates_count += 1
                    continue

                rows.append(row)
                if remove_dup:
                    seen.add(row_tuple)
    except Exception as e:
        logger.warning(f"Ошибка при чтении файла {fpath}: {e}")

    return rows, incomplete_count, duplicates_count, header


# ----------------------------------------------------------------------
# ФУНКЦИЯ КОРРЕКЦИИ ОТРИЦАТЕЛЬНЫХ И \pm INF
# ----------------------------------------------------------------------
def fix_negatives_and_infs(df: pd.DataFrame):
    """
    Корректирует отрицательные значения и ±inf в числовых столбцах:
    - Отрицательные -> заменяем на минимальное положительное (если есть)
    - +inf -> заменяем на max_finite
    - -inf -> заменяем на min_finite
    """
    for col in df.select_dtypes(include=[np.number]).columns:
        finite_values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if finite_values.empty:
            logger.warning(f"Столбец '{col}' содержит только некорректные/пустые значения, пропускаем.")
            continue

        min_positive = finite_values[finite_values > 0].min() if any(finite_values > 0) else None
        max_finite = finite_values.max()
        min_finite = finite_values.min()

        if min_positive:
            df.loc[df[col] < 0, col] = min_positive
        df.loc[df[col] == np.inf, col] = max_finite
        df.loc[df[col] == -np.inf, col] = min_finite


# ---------------------------------------------------------------------
# NormalizeFrame: логика нормализации/логарифма/минмакс
# ---------------------------------------------------------------------
class NormalizeFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.file_list = []

        # Параметры
        self.var_total_lines = tk.StringVar(value="1000000")
        self.var_label = tk.StringVar(value="")
        self.var_shuffle = tk.BooleanVar(value=False)
        self.var_remove_dup = tk.BooleanVar(value=False)
        self.var_remove_incomplete = tk.BooleanVar(value=False)
        self.var_fix_inf = tk.BooleanVar(value=False)    # Корректировать инф/отрицательные
        self.var_do_log = tk.BooleanVar(value=False)     # Логарифм
        self.var_do_minmax = tk.BooleanVar(value=False)  # Min-Max

        # Список чекбоксов для столбцов (при логарифме)
        self.log_columns_vars = {}  # col_name -> tk.BooleanVar()

        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style(self)
        style.configure("TFrame", background="#fafafa")
        style.configure("TLabel", background="#fafafa", foreground="#000", font=("Courier", 12))
        style.configure("TCheckbutton", background="#fafafa", foreground="#000", font=("Courier", 11))
        style.configure("TButton", background="#fafafa", foreground="#000", font=("Courier", 12, "bold"))
        style.map("TButton",
                  background=[("active", "#fafafa")],
                  foreground=[("active", "#000")])
        style.configure("Vertical.TScrollbar", troughcolor="#1E1E1E", background="#dfdfdf")

        lbl_title = ttk.Label(self, text="Нормализовать (объединить) CSV-файлы (параллельно)", font=("Courier", 14, "bold"))
        lbl_title.pack(pady=5)

        # Кнопки "Добавить" и "Очистить"
        top_frame = ttk.Frame(self)
        top_frame.pack(fill="x")
        btn_add = ttk.Button(top_frame, text="Добавить CSV-файлы", command=self.add_files, style="TButton")
        btn_add.pack(side="left", padx=5, pady=5)
        btn_clear = ttk.Button(top_frame, text="Очистить список", command=self.clear_file_list, style="TButton")
        btn_clear.pack(side="left", padx=5, pady=5)

        # Listbox со списком выбранных файлов
        self.listbox = tk.Listbox(self, bg="#dfdfdf", fg="#000", selectbackground="#2d2d2d", height=10)
        self.listbox.pack(fill="both", expand=True, padx=5, pady=5)

        # Фрейм для параметров
        params_frame = ttk.Frame(self)
        params_frame.pack(fill="x", pady=5)

        row_idx = 0
        ttk.Label(params_frame, text="Итоговое кол-во строк:").grid(row=row_idx, column=0, sticky="w", padx=5)
        entry_total = ttk.Entry(params_frame, textvariable=self.var_total_lines, width=10)
        entry_total.grid(row=row_idx, column=1, sticky="w", padx=5)

        row_idx += 1
        ttk.Label(params_frame, text="Метка (label):").grid(row=row_idx, column=0, sticky="w", padx=5)
        entry_label = ttk.Entry(params_frame, textvariable=self.var_label, width=10)
        entry_label.grid(row=row_idx, column=1, sticky="w", padx=5)

        row_idx += 1
        chk_shuffle = ttk.Checkbutton(params_frame, text="Смешать строки (shuffle)", variable=self.var_shuffle)
        chk_shuffle.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=3)

        row_idx += 1
        chk_dup = ttk.Checkbutton(params_frame, text="Удалять дубликаты (между файлами)", variable=self.var_remove_dup)
        chk_dup.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=3)

        row_idx += 1
        chk_inc = ttk.Checkbutton(params_frame, text="Удалять некорректные строки (пустые поля)",
                                  variable=self.var_remove_incomplete)
        chk_inc.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=3)

        # row_idx += 1
        # chk_inf = ttk.Checkbutton(params_frame, text="Корректировать отрицательные и ±inf", variable=self.var_fix_inf)
        # chk_inf.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=3)

        # row_idx += 1
        # chk_log = ttk.Checkbutton(params_frame, text="Применить логарифм (выбрать столбцы)", variable=self.var_do_log,
        #                           command=self.on_log_checkbox)
        # chk_log.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=3)

        # row_idx += 1
        # chk_mm = ttk.Checkbutton(params_frame, text="Min-Max нормализация", variable=self.var_do_minmax)
        # chk_mm.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=3)

        # Фрейм с прокруткой для списка столбцов (логарифм)
        lbl_log_cols = ttk.Label(self, text="Логарифм: выберите числовые столбцы (кроме label):")
        lbl_log_cols.pack(pady=5)
        self.log_cols_frame = ttk.Frame(self)
        self.log_cols_frame.pack(fill="both", expand=True, padx=5, pady=3)

        self.canvas_scroll = tk.Canvas(self.log_cols_frame, bg="#fafafa", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.log_cols_frame, orient="vertical", command=self.canvas_scroll.yview)
        self.canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas_scroll.pack(side="left", fill="both", expand=True)

        self.checkboxes_frame = ttk.Frame(self.canvas_scroll, style="TFrame")
        self.canvas_window = self.canvas_scroll.create_window((0, 0), window=self.checkboxes_frame, anchor="nw")

        def on_frame_configure(event):
            self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all"))
        self.checkboxes_frame.bind("<Configure>", on_frame_configure)

        # По умолчанию скрываем список столбцов (пока чекбокс Логарифм не активен)
        self.log_cols_frame.pack_forget()

        # Прогресс
        self.progress_bar = ttk.Progressbar(self, mode="determinate", length=400, style="Horizontal.TProgressbar")
        self.progress_bar.pack(pady=5)
        self.progress_label = ttk.Label(self, text="Обработка файлов: 0/0")
        self.progress_label.pack(pady=5)

        # Кнопка "Обработать"
        merge_btn = ttk.Button(self, text="Обработать и сохранить", command=self.start_normalize, style="TButton")
        merge_btn.pack(pady=5, anchor="e")

    # Добавление CSV
    def add_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Выберите CSV-файлы",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_paths:
            for f in file_paths:
                if f not in self.file_list:
                    self.file_list.append(f)
                    self.listbox.insert("end", f)
            # После выбора файлов, возможно, нужно обновить список столбцов для логарифма
            self.show_log_column_selection()

    # Очистка списка
    def clear_file_list(self):
        self.file_list.clear()
        self.listbox.delete(0, "end")
        self.log_columns_vars = {}
        for widget in self.checkboxes_frame.winfo_children():
            widget.destroy()

    # Когда пользователь ставит/убирает галочку "Применить логарифм"
    def on_log_checkbox(self):
        if self.var_do_log.get():
            # Показать фрейм со списком столбцов
            self.log_cols_frame.pack(fill="both", expand=True, padx=5, pady=3)
            self.show_log_column_selection()
        else:
            # Спрятать фрейм, очистить чекбоксы
            self.log_cols_frame.pack_forget()
            self.log_columns_vars = {}
            for widget in self.checkboxes_frame.winfo_children():
                widget.destroy()

    # Заполнить список столбцов (только числовые, кроме label)
    def show_log_column_selection(self):
        # Очищаем старые чекбоксы
        for widget in self.checkboxes_frame.winfo_children():
            widget.destroy()
        self.log_columns_vars = {}

        # Если нет файлов -> ничего не делаем
        if not self.file_list or not self.var_do_log.get():
            return

        # Считываем ПЕРВЫЙ файл (для примера) чтобы узнать столбцы
        try:
            sample_df = pd.read_csv(self.file_list[0])
            numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
            # Исключаем label
            if "label" in numeric_cols:
                numeric_cols = numeric_cols.drop("label")

            for col_name in numeric_cols:
                var = tk.BooleanVar()
                cb = ttk.Checkbutton(self.checkboxes_frame, text=col_name, variable=var)
                cb.pack(anchor="w")
                self.log_columns_vars[col_name] = var

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать файл: {e}", parent=self)

    # Начало нормализации
    def start_normalize(self):
        if not self.file_list:
            messagebox.showerror("Ошибка", "Сначала выберите хотя бы один CSV-файл!", parent=self)
            return

        out_path = filedialog.asksaveasfilename(
            title="Сохранить объединённый CSV",
            defaultextension=".csv",
            filetypes=[("CSV File", "*.csv"), ("All Files", "*.*")]
        )
        if not out_path:
            return

        try:
            total_needed = int(self.var_total_lines.get())
            if total_needed < 1:
                messagebox.showerror("Ошибка", "Число итоговых строк должно быть > 0!", parent=self)
                return
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректное число строк!", parent=self)
            return

        thread = Thread(target=self.normalize_thread, args=(out_path, total_needed), daemon=True)
        thread.start()

    def normalize_thread(self, out_path, total_needed):
        do_shuffle = self.var_shuffle.get()
        do_remove_dup = self.var_remove_dup.get()
        do_remove_incomplete = self.var_remove_incomplete.get()
        do_fix_inf = self.var_fix_inf.get()
        do_log = self.var_do_log.get()
        do_minmax = self.var_do_minmax.get()
        label_value = self.var_label.get().strip()

        n_files = len(self.file_list)
        self.set_progress(0, n_files)

        # Параллельно читаем все файлы
        try:
            with multiprocessing.Pool() as pool:
                async_result = pool.starmap_async(
                    process_file,
                    [(fpath, do_remove_incomplete, do_remove_dup) for fpath in self.file_list]
                )
                final_list = async_result.get()
        except Exception as e:
            msg = f"Ошибка при параллельной обработке: {e}"
            logger.error(msg)
            self.after(0, lambda: messagebox.showerror("Ошибка", msg, parent=self))
            return

        self.set_progress(n_files, n_files)

        # Собираем все строки
        all_rows = []
        total_incomplete = 0
        total_duplicates_local = 0
        headers = []
        for (rows, inc_count, dup_count, header) in final_list:
            if header:
                headers.append(header)
            total_incomplete += inc_count
            total_duplicates_local += dup_count
            all_rows.extend(rows)

        main_header = headers[0] if headers else None
        # Дубликаты между файлами
        total_duplicates_global = 0
        if do_remove_dup:
            global_seen = set()
            unique_rows = []
            for row in all_rows:
                rt = tuple(row)
                if rt in global_seen:
                    total_duplicates_global += 1
                    continue
                unique_rows.append(row)
                global_seen.add(rt)
            all_rows = unique_rows

        if len(all_rows) > total_needed:
            all_rows = random.sample(all_rows, total_needed)

        if do_shuffle:
            random.shuffle(all_rows)

        if not main_header and all_rows:
            col_count = len(all_rows[0])
            main_header = [f"col{i+1}" for i in range(col_count)]
        elif not main_header:
            main_header = []

        if not all_rows:
            msg = "В итоге не осталось ни одной строки."
            self.after(0, lambda: messagebox.showinfo("Результат", msg, parent=self))
            return

        df = pd.DataFrame(all_rows, columns=main_header)

        # label, если указано
        if label_value:
            df["label"] = label_value

        # 1) Коррекция ±inf, отрицательных
        if do_fix_inf:
            try:
                fix_negatives_and_infs(df)
            except Exception as e:
                logger.error(f"Ошибка fix_negatives_and_infs: {e}")

        # 2) Логарифмируем выбранные столбцы
        if do_log:
            chosen_cols = [col for col, var in self.log_columns_vars.items() if var.get()]
            for c in chosen_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                    # log1p -> log(1 + x), чтобы избежать log(0)
                    try:
                        df[c] = np.log1p(df[c])
                    except Exception as e:
                        logger.error(f"Ошибка логарифма для столбца '{c}': {e}")

        # 3) Min-Max нормализация (если включено)
        if do_minmax:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for c in numeric_cols:
                if c == "label":
                    continue
                c_min = df[c].min()
                c_max = df[c].max()
                if c_max - c_min != 0:
                    df[c] = (df[c] - c_min) / (c_max - c_min)

        # Сохраняем
        try:
            df.to_csv(out_path, index=False)
        except Exception as e:
            msg = f"Не удалось сохранить итоговый файл: {e}"
            logger.error(msg)
            self.after(0, lambda: messagebox.showerror("Ошибка", msg, parent=self))
            return

        final_count = len(df)

        # Выводим окно с min/max всех числовых столбцов
        def show_min_max_window():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            msg_list = []
            for c in numeric_cols:
                col_min = df[c].min()
                col_max = df[c].max()
                msg_list.append(f"{c}: min={col_min}, max={col_max}")
            window = tk.Toplevel(self)
            window.title("Min/Max после нормализации")
            tk.Label(window, text="Min/Max для числовых столбцов:", font=("Courier", 12, "bold")).pack(pady=5)
            for line in msg_list:
                tk.Label(window, text=line, font=("Courier", 11)).pack(anchor="w", padx=10)

        # Формируем сообщение (лог)
        info_msg = (
            f"Обработка завершена!\n\n"
            f"Файл сохранён: {out_path}\n"
            f"Всего файлов: {len(self.file_list)}\n"
            f"Некорректных строк: {total_incomplete}\n"
            f"Дубликатов (внутри): {total_duplicates_local}\n"
            f"Дубликатов (между): {total_duplicates_global}\n"
            f"Итоговое кол-во строк: {final_count}\n"
            f"Коррекция inf/neg: {'Да' if do_fix_inf else 'Нет'}\n"
            f"Логарифм: {', '.join([c for c,v in self.log_columns_vars.items() if v.get()]) if do_log else 'Нет'}\n"
            f"Min-Max: {'Да' if do_minmax else 'Нет'}\n"
        )

        # Пишем в лог-файл
        file_log_name = os.path.basename(out_path).replace(".csv", "") + ".log"
        file_log_path = os.path.join(LOG_DIR, file_log_name)
        with open(file_log_path, "w", encoding="utf-8") as logf:
            logf.write(info_msg)

        # Показываем итоги
        self.after(0, lambda: messagebox.showinfo("Результат", info_msg, parent=self))
        self.after(0, show_min_max_window)

    def set_progress(self, value, maximum):
        self.after(0, lambda: self._set_progress(value, maximum))

    def _set_progress(self, value, maximum):
        self.progress_bar["maximum"] = maximum
        self.progress_bar["value"] = value
        self.progress_label.config(text=f"Обработка файлов: {value}/{maximum}")


# --------------------------------------------------------------------
# Функция для вкладки "Анализ": выводит min/max выбранных столбцов
# --------------------------------------------------------------------
def find_min_max_for_selected_columns(df: pd.DataFrame):
    numeric_columns = df.select_dtypes(include=['number']).columns
    results = []
    for column in numeric_columns:
        min_value = df[column].min()
        max_value = df[column].max()
        results.append({"Column": column, "Min": min_value, "Max": max_value})
    return pd.DataFrame(results)


# --------------------------------------------------------------------
# AnalyzeFrame — вкладка анализа (графики)
# --------------------------------------------------------------------
class AnalyzeFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.file_path = None
        self.df = None
        self.column_vars = {}
        self.canvas = None
        self.select_all_button = None
        self.plot_button = None
        self.min_max_button = None
        self.save_graph_button = None
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style(self)
        style.configure("TFrame", background="#fafafa")
        style.configure("TLabel", background="#fafafa", foreground="#000", font=("Courier", 12))
        style.configure("TButton", background="#fafafa", foreground="#000", font=("Courier", 12, "bold"))
        style.map("TButton",
                  background=[("active", "#fafafa")],
                  foreground=[("active", "#000")])
        style.configure("Horizontal.TProgressbar", troughcolor="#fafafa", background="#000", thickness=25)
        style.configure("Vertical.TScrollbar", troughcolor="#fafafa", background="#000")

        lbl_title = ttk.Label(self, text="Анализ данных CSV", font=("Courier", 14, "bold"))
        lbl_title.pack(pady=5)

        btn_add = ttk.Button(self, text="Выбрать CSV-файл", command=self.select_file, style="TButton")
        btn_add.pack(pady=5)

        self.progress_bar = ttk.Progressbar(self, mode="determinate", length=400, style="Horizontal.TProgressbar")
        self.progress_bar.pack(pady=5)

        self.progress_label = ttk.Label(self, text="Ожидание...")
        self.progress_label.pack(pady=5)

        center_frame = ttk.Frame(self)
        center_frame.pack(fill="both", expand=True, pady=5)

        self.canvas_scroll = tk.Canvas(center_frame, bg="#fafafa", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(center_frame, orient="vertical", command=self.canvas_scroll.yview)
        self.canvas_scroll.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas_scroll.pack(side="left", fill="both", expand=True)

        self.checkboxes_frame = ttk.Frame(self.canvas_scroll, style="TFrame")
        self.canvas_window = self.canvas_scroll.create_window((0, 0), window=self.checkboxes_frame, anchor="nw")

        def on_frame_configure(event):
            self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all"))
        self.checkboxes_frame.bind("<Configure>", on_frame_configure)

        self.graph_frame = ttk.Frame(self)
        self.graph_frame.pack(fill="both", expand=True, pady=5)

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите CSV-файл",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path = file_path
            self.progress_label.config(text=f"Файл выбран: {os.path.basename(file_path)}")
            self.load_columns()

    def load_columns(self):
        try:
            self.df = pd.read_csv(self.file_path)
            numeric_columns = self.df.select_dtypes(include=['number']).columns

            for widget in self.checkboxes_frame.winfo_children():
                widget.destroy()

            if self.select_all_button:
                self.select_all_button.destroy()

            # Разделим все чекбоксы на несколько колонок
            num_columns = 3
            column_frames = []
            for i in range(num_columns):
                frame = ttk.Frame(self.checkboxes_frame)
                frame.grid(row=0, column=i, sticky="nsew", padx=5)
                column_frames.append(frame)

            self.column_vars = {}
            idx = 0
            for column in numeric_columns:
                var = tk.BooleanVar()
                cb = ttk.Checkbutton(column_frames[idx % num_columns], text=column, variable=var,
                                     command=self.update_buttons_state)
                cb.pack(anchor="w")
                self.column_vars[column] = var
                idx += 1

            # Кнопка "Выбрать все"
            self.select_all_button = ttk.Button(
                self.checkboxes_frame,
                text="Выбрать все",
                command=self.select_all_columns,
                style="TButton"
            )
            self.select_all_button.grid(row=1, column=0, columnspan=num_columns, pady=5)

            self.create_action_buttons()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}", parent=self)

    def create_action_buttons(self):
        if self.plot_button:
            self.plot_button.destroy()
        if self.min_max_button:
            self.min_max_button.destroy()

        self.plot_button = ttk.Button(
            self,
            text="Построить график",
            command=self.start_analysis,
            style="TButton",
            state="disabled"
        )
        self.plot_button.pack(pady=5)

        self.min_max_button = ttk.Button(
            self,
            text="Найти минимум/максимум",
            command=self.find_min_max,
            style="TButton",
            state="disabled"
        )
        self.min_max_button.pack(pady=5)

    def update_buttons_state(self):
        selected_columns = [col for col, var in self.column_vars.items() if var.get()]
        if selected_columns:
            self.plot_button.config(state="normal")
            self.min_max_button.config(state="normal")
        else:
            self.plot_button.config(state="disabled")
            self.min_max_button.config(state="disabled")

    def select_all_columns(self):
        new_state = not all(var.get() for var in self.column_vars.values())
        for var in self.column_vars.values():
            var.set(new_state)
        self.update_buttons_state()

    def start_analysis(self):
        if self.df is None:
            messagebox.showerror("Ошибка", "Сначала выберите CSV-файл!", parent=self)
            return
        selected_columns = [col for col, var in self.column_vars.items() if var.get()]
        if not selected_columns:
            messagebox.showerror("Ошибка", "Выберите хотя бы одну колонку для анализа!", parent=self)
            return
        if len(selected_columns) > 5:
            messagebox.showerror("Ошибка", "Можно выбрать не более 5 колонок для построения графика!", parent=self)
            return

        if hasattr(self, "canvas") and self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(8, 4))
        for column in selected_columns:
            x_values = range(1, len(self.df) + 1)
            y_values = self.df[column].tolist()
            ax.plot(x_values, y_values, marker="o", linestyle="-", markersize=2, label=column)

        ax.set_xlabel("Номер строки")
        ax.set_ylabel("Значения")
        ax.set_title("График значений выбранных столбцов")
        ax.legend()
        ax.grid(True)

        self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        if not self.save_graph_button:
            self.save_graph_button = ttk.Button(
                self,
                text="Сохранить график",
                command=self.save_graph,
                style="TButton"
            )
            self.save_graph_button.pack(pady=5)

    def find_min_max(self):
        selected_columns = [col for col, var in self.column_vars.items() if var.get()]
        if not selected_columns:
            messagebox.showerror("Ошибка", "Выберите хотя бы одну колонку!", parent=self)
            return
        result_df = find_min_max_for_selected_columns(self.df[selected_columns])
        # Покажем окно с результатами
        window = tk.Toplevel(self)
        window.title("Min/Max для выбранных столбцов")
        tk.Label(window, text="Результат:", font=("Courier", 12, "bold")).pack(pady=5)
        for idx, row in result_df.iterrows():
            line = f"{row['Column']}: min={row['Min']}, max={row['Max']}"
            tk.Label(window, text=line, font=("Courier", 11)).pack(anchor="w", padx=10)

    def save_graph(self):
        if not self.canvas:
            messagebox.showerror("Ошибка", "График ещё не построен!", parent=self)
            return
        file_path = filedialog.asksaveasfilename(
            title="Сохранить график",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            self.canvas.figure.savefig(file_path)
            messagebox.showinfo("Готово!", f"График сохранён: {file_path}", parent=self)


# ---------------------------------------------------------------------
# Главное окно приложения
# ---------------------------------------------------------------------
class MainApp(ThemedTk):
    def __init__(self):
        super().__init__()
        self.set_theme("arc")
        self.title("Приложение: Скачивание и Нормализация CSV (Parallel)")
        self.geometry("900x700")
        self.configure(background="#fafafa")

        style = ttk.Style()
        style.configure('TFrame', background='#fafafa')
        style.configure('TLabel', background='#fafafa', foreground='#000', font=('Segoe UI', 10))
        style.configure('TButton', background='#fafafa', foreground='#000', font=('Segoe UI', 10, 'bold'))
        style.map('TButton',
                  background=[('active', '#fafafa')],
                  foreground=[('active', '#000')])
        style.configure('TCheckbutton', background='#fafafa', foreground='#000', font=('Segoe UI', 10))
        style.configure('Horizontal.TProgressbar', troughcolor='#fafafa', background='#000', thickness=25)
        style.configure('Vertical.TScrollbar', troughcolor='#fafafa', background='#000')

        all_folders = [folder for folder in get_links(BASE_URL) if folder.endswith('/')]

        self.download_frame = DownloadFrame(self, all_folders)
        self.normalize_frame = NormalizeFrame(self)
        self.analyze_frame = AnalyzeFrame(self)

        menubar = tk.Menu(self, background="#fafafa", foreground="#000", font=('Segoe UI', 10))
        self.config(menu=menubar)

        menu_functions = tk.Menu(menubar, tearoff=False, background="#fafafa", foreground="#000", font=('Segoe UI', 10))
        menu_functions.add_command(label="Скачать датасет", command=self.show_download)
        menu_functions.add_command(label="Нормализовать датасет", command=self.show_normalize)
        menu_functions.add_command(label="Анализировать датасет", command=self.show_analyze)
        menu_functions.add_separator()
        menu_functions.add_command(label="Выход", command=self.quit)
        menubar.add_cascade(label="Меню", menu=menu_functions)

        self.show_download()

    def show_download(self):
        self.normalize_frame.pack_forget()
        self.analyze_frame.pack_forget()
        self.download_frame.pack(fill="both", expand=True)

    def show_normalize(self):
        self.download_frame.pack_forget()
        self.analyze_frame.pack_forget()
        self.normalize_frame.pack(fill="both", expand=True)

    def show_analyze(self):
        self.download_frame.pack_forget()
        self.normalize_frame.pack_forget()
        self.analyze_frame.pack(fill="both", expand=True)


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
