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

# ------------------------------------------------
# Установка зависимостей
# pip install requests numpy pandas matplotlib beautifulsoup4 ttkthemes 
# ------------------------------------------------


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
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return [a["href"] for a in soup.find_all("a", href=True)]


def download_file(file_url, file_path):
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
# DownloadFrame (скачивание)
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
        style.configure("Vertical.TScrollbar", troughcolor="#fafafa", background="#dfdfdfdfdfdf")

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
            style="TButton"  # Явно указываем стиль
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


# ----------------------------------------------------------
# ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА ОТДЕЛЬНОГО ФАЙЛА
# ----------------------------------------------------------
def process_file(
    fpath: str,
    remove_incomplete: bool,
    remove_dup: bool
):
    """
    Запускается в отдельном процессе (multiprocessing).
    Построчно читает один CSV-файл:
      - если remove_incomplete=True -> пропускает строки с пустыми полями
      - если remove_dup=True -> пропускает строки, которые уже встретились в этом файле
    Логирует каждый пропуск (как раньше).
    Возвращает кортеж:
       (rows, incomplete_count, duplicates_count, header)
    где:
       rows = список строк (без заголовка)
       incomplete_count = сколько строк пропущено по причине пустых полей
       duplicates_count = сколько строк пропущено как дубликаты
       header = первая строка файла
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
                # Первый ряд - заголовок
                if row_index == 1:
                    header = row
                    continue

                # Проверяем некорректность
                if remove_incomplete and any(col.strip() == "" for col in row):
                    logger.info(f"[Некорректная строка] File='{fpath}', Line={row_index}, Data={row}")
                    incomplete_count += 1
                    continue

                # Проверяем дубликат
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


# --------------------------------------------------------------------------
# ЛОГИКА КОРРЕКЦИИ ОТРИЦАТЕЛЬНЫХ И \pm INF
# --------------------------------------------------------------------------
def fix_negatives_and_infs(df: pd.DataFrame):
    """
    Корректирует отрицательные значения и ±inf в числовых столбцах:
    - Отрицательные значения заменяются на минимальное положительное значение в столбце
    - +inf заменяется на максимальное конечное значение в столбце
    - -inf заменяется на минимальное конечное значение в столбце
    """
    for col in df.select_dtypes(include=[np.number]).columns:
        finite_values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        
        if finite_values.empty:
            logger.warning(f"Столбец '{col}' содержит только некорректные значения")
            continue
        
        min_positive = finite_values[finite_values > 0].min() if any(finite_values > 0) else None
        max_finite = finite_values.max()
        min_finite = finite_values.min()
        
        # Замена значений
        if min_positive:
            df.loc[df[col] < 0, col] = min_positive
        df.loc[df[col] == np.inf, col] = max_finite
        df.loc[df[col] == -np.inf, col] = min_finite


# --------------------------------------------------------------------------
#           NormalizeFrame: параллельная обработка КАЖДОГО файла
# --------------------------------------------------------------------------
class NormalizeFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)

        self.file_list = []
        self.var_total_lines = tk.StringVar(value="1000000")
        self.var_label = tk.StringVar(value="")
        self.var_shuffle = tk.BooleanVar(value=False)
        self.var_remove_dup = tk.BooleanVar(value=False)
        self.var_remove_incomplete = tk.BooleanVar(value=False)
        self.var_fix_inf_and_negatives = tk.BooleanVar(value=False)
        self.var_min_max_normalize = tk.BooleanVar(value=False)  # Новый чекбокс для Min-Max

        # Чекбоксы для выбора столбцов для логарифмического преобразования
        self.log_columns_vars = {}
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

        top_frame = ttk.Frame(self)
        top_frame.pack(fill="x")

        btn_add = ttk.Button(top_frame, text="Добавить CSV-файлы", command=self.add_files, style="TButton")
        btn_add.pack(side="left", padx=5, pady=5)

        btn_clear = ttk.Button(top_frame, text="Очистить список", command=self.clear_file_list, style="TButton")
        btn_clear.pack(side="left", padx=5, pady=5)

        self.listbox = tk.Listbox(self, bg="#dfdfdf", fg="#000", selectbackground="#2d2d2d", height=10)
        self.listbox.pack(fill="both", expand=True, padx=5, pady=5)

        # Заголовок для выбора столбцов
        lbl_log_columns = ttk.Label(self, text="Выберите столбцы для логарифмического преобразования:", font=("Courier", 12, "bold"))
        lbl_log_columns.pack(pady=5)

        # Фрейм для чекбоксов с прокруткой
        log_columns_frame = ttk.Frame(self)
        log_columns_frame.pack(fill="both", expand=True, pady=5)

        self.canvas_scroll = tk.Canvas(log_columns_frame, bg="#fafafa", highlightthickness=0)
        scrollbar = ttk.Scrollbar(log_columns_frame, orient="vertical", command=self.canvas_scroll.yview)
        self.canvas_scroll.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.canvas_scroll.pack(side="left", fill="both", expand=True)

        self.checkboxes_frame = ttk.Frame(self.canvas_scroll, style="TFrame")
        self.canvas_window = self.canvas_scroll.create_window((0, 0), window=self.checkboxes_frame, anchor="nw")

        def on_frame_configure(event):
            self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all"))

        self.checkboxes_frame.bind("<Configure>", on_frame_configure)

        params_frame = ttk.Frame(self)
        params_frame.pack(fill="x", pady=5)

        lbl_total_lines = ttk.Label(params_frame, text="Итоговое кол-во строк (после очистки):")
        lbl_total_lines.grid(row=0, column=0, sticky="w", padx=5)
        entry_total_lines = ttk.Entry(params_frame, textvariable=self.var_total_lines, width=12)
        entry_total_lines.grid(row=0, column=1, sticky="w", padx=5)

        lbl_label = ttk.Label(params_frame, text="Метка (label):")
        lbl_label.grid(row=1, column=0, sticky="w", padx=5)
        entry_label = ttk.Entry(params_frame, textvariable=self.var_label, width=12)
        entry_label.grid(row=1, column=1, sticky="w", padx=5)

        chk_shuffle = ttk.Checkbutton(params_frame, text="Смешать строки (Random Shuffle)", variable=self.var_shuffle)
        chk_shuffle.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=3)

        chk_dup = ttk.Checkbutton(params_frame, text="Удалять дубликаты (между всеми файлами)", variable=self.var_remove_dup)
        chk_dup.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=3)

        chk_inc = ttk.Checkbutton(params_frame, text="Удалять некорректные строки (пустые поля)",
                                  variable=self.var_remove_incomplete)
        chk_inc.grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=3)

        chk_fix_inf_neg = ttk.Checkbutton(params_frame,
                                          text="Корректировать отрицательные и ±inf",
                                          variable=self.var_fix_inf_and_negatives)
        chk_fix_inf_neg.grid(row=5, column=0, columnspan=2, sticky="w", padx=5, pady=3)

        # Новый чекбокс для Min-Max нормализации
        chk_min_max = ttk.Checkbutton(params_frame,
                                      text="Применить Min-Max нормализацию",
                                      variable=self.var_min_max_normalize)
        chk_min_max.grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=3)

        # Прогресс (по кол-ву ФАЙЛОВ)
        self.progress_bar = ttk.Progressbar(self, mode="determinate", length=400, style="Horizontal.TProgressbar")
        self.progress_bar.pack(pady=5)
        self.progress_label = ttk.Label(self, text="Обработка файлов: 0/0")
        self.progress_label.pack(pady=5)

        merge_btn = ttk.Button(self, text="Обработать и сохранить", command=self.start_normalize, style="TButton")
        merge_btn.pack(pady=5, anchor="e")

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
            self.show_log_column_selection()

    def show_log_column_selection(self):
        # Очищаем предыдущие чекбоксы
        for widget in self.checkboxes_frame.winfo_children():
            widget.destroy()

        # Загружаем первый файл для получения списка столбцов
        if self.file_list:
            try:
                sample_df = pd.read_csv(self.file_list[0])
                numeric_columns = sample_df.select_dtypes(include=['number']).columns

                self.log_columns_vars = {}
                for i, column in enumerate(numeric_columns):
                    var = tk.BooleanVar()
                    cb = ttk.Checkbutton(self.checkboxes_frame, text=column, variable=var)
                    cb.grid(row=i, column=0, sticky="w", padx=5)
                    self.log_columns_vars[column] = var

            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось прочитать файл для получения столбцов: {e}", parent=self)

    def clear_file_list(self):
        self.file_list.clear()
        self.listbox.delete(0, "end")
        for widget in self.checkboxes_frame.winfo_children():
            widget.destroy()

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

        # Запускаем в отдельном потоке, чтобы GUI не завис
        thread = Thread(target=self.normalize_thread, args=(out_path, total_needed), daemon=True)
        thread.start()
    def normalize_thread(self, out_path, total_needed):
        """
        Основной метод: ПАРАЛЛЕЛЬНО обрабатывает каждый файл (process_file),
        потом сливает результаты. Если do_remove_dup=True, удаляем дубликаты
        и МЕЖДУ файлами.
        """
        do_shuffle = self.var_shuffle.get()
        do_remove_dup = self.var_remove_dup.get()
        do_remove_incomplete = self.var_remove_incomplete.get()
        do_fix_inf_neg = self.var_fix_inf_and_negatives.get()
        do_min_max_normalize = self.var_min_max_normalize.get()  # Новая переменная для Min-Max
        label_value = self.var_label.get().strip()

        # Получаем выбранные поля для логарифмического преобразования
        log_columns = [col for col, var in self.log_columns_vars.items() if var.get()]

        n_files = len(self.file_list)
        self.set_progress(0, n_files)

        # Пул процессов для process_file
        results = []
        try:
            with multiprocessing.Pool() as pool:
                async_result = pool.starmap_async(
                    process_file,
                    [(fpath, do_remove_incomplete, do_remove_dup) for fpath in self.file_list]
                )
                final_list = async_result.get()

            # Все файлы готовы:
            self.set_progress(n_files, n_files)

            results = final_list
        except Exception as e:
            msg = f"Ошибка при параллельной обработке: {e}"
            logger.error(msg)
            self.after(0, lambda: messagebox.showerror("Ошибка", msg, parent=self))
            return

        # Теперь объединим результаты
        all_rows = []
        total_incomplete = 0
        total_duplicates_local = 0
        headers = []
        for i, (rows, inc_count, dup_count, header) in enumerate(results, start=1):
            if header:
                headers.append(header)
            total_incomplete += inc_count
            total_duplicates_local += dup_count
            all_rows.extend(rows)

        # Выберем "первый" заголовок как основной
        main_header = headers[0] if headers else None

        # Если включено remove_dup => ещё убираем дубликаты МЕЖДУ файлами
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

        # Если строк больше, чем требуется
        if len(all_rows) > total_needed:
            all_rows = random.sample(all_rows, total_needed)

        # Перемешать, если нужно
        if do_shuffle:
            random.shuffle(all_rows)

        # Если заголовка нет, пытаемся создать
        if not main_header and all_rows:
            col_count = len(all_rows[0])
            main_header = [f"col{i+1}" for i in range(col_count)]
        elif not main_header:
            main_header = []

        if not all_rows:
            msg = "В итоге не осталось ни одной строки (все удалены?)."
            self.after(0, lambda: messagebox.showinfo("Результат", msg, parent=self))
            return

        # Собираем DataFrame
        df = pd.DataFrame(all_rows, columns=main_header)

        # Если задана метка (label)
        if label_value:
            df["label"] = label_value

        # --- ВАЖНО: Корректируем минусы и inf, если включен чекбокс ---
        if do_fix_inf_neg:
            try:
                fix_negatives_and_infs(df)
            except ValueError as e:
                msg = f"Ошибка коррекции значений: {e}"
                logger.error(msg)
                self.after(0, lambda: messagebox.showerror("Ошибка", msg, parent=self))
                return

        # Применяем логарифмическое преобразование к выбранным полям
        for column in log_columns:
            if column in df.columns:
                try:
                    # Преобразуем столбец в числовой тип, игнорируя ошибки
                    df[column] = pd.to_numeric(df[column], errors="coerce")
                    # Применяем логарифмическое преобразование
                    df[column] = np.log1p(df[column])
                except Exception as e:
                    msg = f"Ошибка при логарифмическом преобразовании столбца '{column}': {e}"
                    logger.error(msg)
                    self.after(0, lambda: messagebox.showerror("Ошибка", msg, parent=self))
                    return

        # Выполняем Min-Max нормализацию, если включен чекбокс
        if do_min_max_normalize:
            numeric_columns = df.select_dtypes(include=['number']).columns
            for column in numeric_columns:
                min_val = df[column].min()
                max_val = df[column].max()
                if max_val - min_val > 0:
                    df[column] = (df[column] - min_val) / (max_val - min_val)

        # Пытаемся сохранить
        try:
            df.to_csv(out_path, index=False)
        except Exception as e:
            msg = f"Не удалось сохранить итоговый файл: {e}"
            logger.error(msg)
            self.after(0, lambda: messagebox.showerror("Ошибка", msg, parent=self))
            return

        final_count = len(df)
        file_name = f"logs/{os.path.basename(out_path).replace('.csv', '')}.log"
        msg = (
            f"Обработка завершена!\n\n"
            f"Файл сохранён: {out_path}\n"
            f"Всего файлов: {len(self.file_list)}\n\n"
            f"Некорректных строк (по файлам): {total_incomplete}\n"
            f"Дубликатов (внутри файлов): {total_duplicates_local}\n"
            f"Дубликатов (между файлами): {total_duplicates_global}\n"
            f"Итоговое кол-во строк: {final_count} (с учётом лимита = {total_needed})\n"
            f"Коррекция отрицательных/inf: {'Да' if do_fix_inf_neg else 'Нет'}\n"
            f"Логарифмическое преобразование: {', '.join(log_columns) if log_columns else 'Нет'}\n"
            f"Min-Max нормализация: {'Да' if do_min_max_normalize else 'Нет'}"
        )
        with open(file_name, 'w', encoding="utf-8") as file:
            file.write(msg)

        self.after(0, lambda: messagebox.showinfo("Результат", msg, parent=self))

    # ----------------- Прогресс по КОЛ-ВУ ФАЙЛОВ -----------------
    def set_progress(self, value, maximum):
        self.after(0, lambda: self._set_progress(value, maximum))

    def _set_progress(self, value, maximum):
        self.progress_bar["maximum"] = maximum
        self.progress_bar["value"] = value
        self.progress_label.config(text=f"Обработка файлов: {value}/{maximum}")


def find_min_max_for_selected_columns(df: pd.DataFrame):
    """
    Функция находит минимальное и максимальное значения для всех числовых колонок в DataFrame.

    :param df: DataFrame с данными
    :return: DataFrame с колонками: "Column", "Min", "Max"
    """
    numeric_columns = df.select_dtypes(include=['number']).columns  # Берем только числовые колонки
    results = []

    for column in numeric_columns:
        min_value = df[column].min()
        max_value = df[column].max()
        results.append({"Column": column, "Min": min_value, "Max": max_value})

    return pd.DataFrame(results)

class AnalyzeFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.file_path = None
        self.df = None
        self.column_vars = {}
        self.canvas = None  # Для встроенного графика
        self.select_all_button = None  # Кнопка "Выбрать все"
        self.plot_button = None  # Кнопка "Построить график"
        self.min_max_button = None  # Кнопка "Найти минимум/максимум"
        self.save_graph_button = None  # Кнопка "Сохранить график"
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

        # Центральный контейнер для чекбоксов с прокруткой
        center_frame = ttk.Frame(self)
        center_frame.pack(fill="both", expand=True, pady=5)

        self.canvas_scroll = tk.Canvas(center_frame, bg="#fafafa", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(center_frame, orient="vertical", command=self.canvas_scroll.yview)
        self.canvas_scroll.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas_scroll.pack(side="left", fill="both", expand=True)

        self.checkboxes_frame = ttk.Frame(self.canvas_scroll, style="TFrame")
        self.canvas_window = self.canvas_scroll.create_window((0, 0), window=self.checkboxes_frame, anchor="center")

        def on_frame_configure(event):
            self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all"))

        self.checkboxes_frame.bind("<Configure>", on_frame_configure)

        # Место для графика
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
            # Чтение файла и получение числовых колонок
            self.df = pd.read_csv(self.file_path)
            numeric_columns = self.df.select_dtypes(include=['number']).columns

            # Очистка предыдущих чекбоксов
            for widget in self.checkboxes_frame.winfo_children():
                widget.destroy()

            # Удаление старой кнопки "Выбрать все", если она существует
            if self.select_all_button:
                self.select_all_button.destroy()

            # Разделение чекбоксов на несколько столбцов
            num_columns = 3  # Количество столбцов
            column_frames = []
            for i in range(num_columns):
                frame = ttk.Frame(self.checkboxes_frame)
                frame.grid(row=0, column=i, sticky="nsew", padx=5)
                column_frames.append(frame)

            self.column_vars = {}
            for i, column in enumerate(numeric_columns):
                var = tk.BooleanVar()
                cb = ttk.Checkbutton(column_frames[i % num_columns], text=column, variable=var, command=self.update_buttons_state)
                cb.pack(anchor="w")
                self.column_vars[column] = var

            # Добавление кнопки "Выбрать все" внизу списка
            self.select_all_button = ttk.Button(
                self.checkboxes_frame,
                text="Выбрать все",
                command=self.select_all_columns,
                style="TButton"
            )
            self.select_all_button.grid(row=1, column=0, columnspan=num_columns, pady=5)

            # Создание кнопок "Построить график" и "Найти минимум/максимум"
            self.create_action_buttons()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {e}", parent=self)

    def create_action_buttons(self):
        # Удаление старых кнопок, если они существуют
        if self.plot_button:
            self.plot_button.destroy()
        if self.min_max_button:
            self.min_max_button.destroy()

        # Создание новых кнопок
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
        # Проверка состояния чекбоксов
        selected_columns = [col for col, var in self.column_vars.items() if var.get()]
        if selected_columns:
            self.plot_button.config(state="normal")
            self.min_max_button.config(state="normal")
        else:
            self.plot_button.config(state="disabled")
            self.min_max_button.config(state="disabled")

    def select_all_columns(self):
        # Переключение состояния всех чекбоксов
        new_state = not all(var.get() for var in self.column_vars.values())
        for var in self.column_vars.values():
            var.set(new_state)
        self.update_buttons_state()

    def start_analysis(self):
        if self.df is None:
            messagebox.showerror("Ошибка", "Сначала выберите CSV-файл!", parent=self)
            return

        # Получение выбранных колонок
        selected_columns = [col for col, var in self.column_vars.items() if var.get()]
        if not selected_columns:
            messagebox.showerror("Ошибка", "Выберите хотя бы одну колонку для анализа!", parent=self)
            return

        if len(selected_columns) > 5:
            messagebox.showerror("Ошибка", "Можно выбрать не более 5 колонок для построения графика!", parent=self)
            return

        # Очистка предыдущего графика
        if hasattr(self, "canvas") and self.canvas:
            self.canvas.get_tk_widget().destroy()

        # Построение графиков
        fig, ax = plt.subplots(figsize=(8, 4))
        for column in selected_columns:
            x_values = range(1, len(self.df) + 1)
            y_values = self.df[column].tolist()
            ax.plot(x_values, y_values, marker="o", linestyle="-", markersize=2, label=column)

        ax.set_xlabel("Номер строки (Row Count)")
        ax.set_ylabel("Значения")
        ax.set_title("График значений выбранных колонок")
        ax.legend()
        ax.grid(True)

        # Встраивание графика в интерфейс
        self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Добавление кнопки "Сохранить график"
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

        # Вывод результата в консоль
        print(result_df)

        # Сохранение результатов в лог
        log_dir = "logs/app/min_max/"
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/min_max_results.log", "w", encoding="utf-8") as f:
            f.write(result_df.to_string())

        # Сохранение результатов в CSV
        # result_df.to_csv("min_max_results.csv", index=False)

        messagebox.showinfo("Готово!", "Минимум и максимум найдены! Результаты сохранены.", parent=self)

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
# --------------------------------------------------------------------------
#                         Главное окно
# --------------------------------------------------------------------------
class MainApp(ThemedTk):
    def __init__(self):
        super().__init__()
        
        # Устанавливаем современную тему
        self.set_theme("arc")  # arc - одна из доступных тем
        
        self.title("Приложение: Скачивание и Нормализация CSV (Parallel)")
        self.geometry("800x600")
        self.configure(background="#fafafa")  # Фон окна тоже делаем черным
        
        # Настройка стилей
        style = ttk.Style()
        style.configure('TFrame', background='#fafafa')  # Фон фреймов
        style.configure('TLabel', background='#fafafa', foreground='#000', font=('Segoe UI', 10))  # Цвет текста меток
        style.configure('TButton', background='#fafafa', foreground='#000', font=('Segoe UI', 10, 'bold'))  # Кнопки
        style.map('TButton',
                  background=[('active', '#fafafa')],  # Цвет при наведении
                  foreground=[('active', '#000')])  # Цвет текста при наведении
        style.configure('TCheckbutton', background='#fafafa', foreground='#000', font=('Segoe UI', 10))  # Чекбоксы
        style.configure('Horizontal.TProgressbar', troughcolor='#fafafa', background='#000', thickness=25)  # Прогресс-бар
        style.configure('Vertical.TScrollbar', troughcolor='#fafafa', background='#000')  # Скроллбар
        
        # Получаем список папок (для скачивания)
        all_folders = [folder for folder in get_links(BASE_URL) if folder.endswith('/')]

        # Создаем фреймы для разных экранов
        self.download_frame = DownloadFrame(self, all_folders)
        self.normalize_frame = NormalizeFrame(self)
        self.analyze_frame = AnalyzeFrame(self)

        # Создаем меню
        menubar = tk.Menu(self, background="#fafafa", foreground="#000", font=('Segoe UI', 10))
        self.config(menu=menubar)

        menu_functions = tk.Menu(menubar, tearoff=False, background="#fafafa", foreground="#000", font=('Segoe UI', 10))
        menu_functions.add_command(label="Скачать датасет", command=self.show_download)
        menu_functions.add_command(label="Нормализовать датасет", command=self.show_normalize)
        menu_functions.add_command(label="Анализировать датасет", command=self.show_analyze)
        menu_functions.add_command(label="Выход", command=self.quit)
        menubar.add_cascade(label="Меню", menu=menu_functions)

        # menu_file = tk.Menu(menubar, tearoff=False, background="#fafafa", foreground="#000", font=('Segoe UI', 10))
        # menubar.add_cascade(label="Файл", menu=menu_file)

        # Отображаем начальный экран
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
