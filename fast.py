import os
import requests
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import pandas as pd
import csv
import logging
import random

BASE_URL = "http://205.174.165.80/IOTDataset/CIC_IOT_Dataset2023/Dataset/CSV/CSV/"
SAVE_DIR = "downloads"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_DIR = "logs"
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
# DownloadFrame (скачивание) - без изменений
# ------------------------------------------------
class DownloadFrame(ttk.Frame):
    def __init__(self, master, all_folders):
        super().__init__(master, padding=10)
        self.all_folders = all_folders
        self.folder_vars = {}
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style(self)
        style.configure("TFrame", background="#121212")
        style.configure("TLabel", background="#121212", foreground="cyan", font=("Courier", 12))
        style.configure("TCheckbutton", background="#121212", foreground="cyan", font=("Courier", 11))
        style.configure("TButton", background="black", foreground="white", font=("Courier", 12, "bold"))
        style.map("TButton",
                  background=[("active", "#2d2d2d")],
                  foreground=[("active", "white")])
        style.configure("Horizontal.TProgressbar",
                        troughcolor="black",
                        bordercolor="black",
                        background="cyan",
                        lightcolor="cyan",
                        darkcolor="cyan")
        style.configure("Vertical.TScrollbar", troughcolor="#1E1E1E", background="#333333")

        title_label = ttk.Label(self, text="Выберите папки для скачивания:", font=("Courier", 14, "bold"))
        title_label.pack(pady=5)

        self.status_label = ttk.Label(self, text=f"Доступно папок: {len(self.all_folders)} | Выбрано: 0")
        self.status_label.pack(pady=5)

        checkboxes_frame = ttk.Frame(self)
        checkboxes_frame.pack(fill="both", expand=True, pady=5)

        canvas_bg_color = "#1E1E1E"
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

        download_btn = ttk.Button(self, text="Скачать выбранные", command=self.start_download)
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
#   ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА ОТДЕЛЬНОГО ФАЙЛА
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

        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style(self)
        style.configure("TFrame", background="#121212")
        style.configure("TLabel", background="#121212", foreground="cyan", font=("Courier", 12))
        style.configure("TCheckbutton", background="#121212", foreground="cyan", font=("Courier", 11))
        style.configure("TButton", background="black", foreground="white", font=("Courier", 12, "bold"))
        style.map("TButton",
                  background=[("active", "#2d2d2d")],
                  foreground=[("active", "white")])
        style.configure("Vertical.TScrollbar", troughcolor="#1E1E1E", background="#333333")

        lbl_title = ttk.Label(self, text="Нормализовать (объединить) CSV-файлы (параллельно)", font=("Courier", 14, "bold"))
        lbl_title.pack(pady=5)

        top_frame = ttk.Frame(self)
        top_frame.pack(fill="x")

        btn_add = ttk.Button(top_frame, text="Добавить CSV-файлы", command=self.add_files)
        btn_add.pack(side="left", padx=5, pady=5)

        btn_clear = ttk.Button(top_frame, text="Очистить список", command=self.clear_file_list)
        btn_clear.pack(side="left", padx=5, pady=5)

        self.listbox = tk.Listbox(self, bg="#333333", fg="cyan", selectbackground="#2d2d2d", height=10)
        self.listbox.pack(fill="both", expand=True, padx=5, pady=5)

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

        # Прогресс (по кол-ву ФАЙЛОВ)
        self.progress_bar = ttk.Progressbar(self, mode="determinate", length=400, style="Horizontal.TProgressbar")
        self.progress_bar.pack(pady=5)
        self.progress_label = ttk.Label(self, text="Обработка файлов: 0/0")
        self.progress_label.pack(pady=5)

        merge_btn = ttk.Button(self, text="Обработать и сохранить", command=self.start_normalize)
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

    def clear_file_list(self):
        self.file_list.clear()
        self.listbox.delete(0, "end")

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
        label_value = self.var_label.get().strip()

        n_files = len(self.file_list)
        self.set_progress(0, n_files)

        # Запускаем пул процессов (примерно кол-во CPU, либо зашить 4 и т.д.)
        # В каждом процессе вызовем process_file(...)
        results = []
        def update_file_progress(idx):
            self.set_progress(idx, n_files)

        try:
            with multiprocessing.Pool() as pool:
                # starmap_async чтобы иметь возможность отслеживать прогресс
                async_result = pool.starmap_async(
                    process_file,
                    [(fpath, do_remove_incomplete, do_remove_dup) for fpath in self.file_list]
                )

                # Ожидаем результаты, параллельно можем считать прогресс
                # Но starmap_async даёт нам всё разом, без "chunk-wise" колбэков.
                # Поэтому просто дождёмся .get() и тогда обновим progress
                final_list = async_result.get()

            # final_list = [ (rows, inc_count, dup_count, header), ... ] по каждому файлу
            # Обновим прогресс: все файлы готовы
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
        headers = []  # соберём все заголовки (на случай, если разные)
        for i, (rows, inc_count, dup_count, header) in enumerate(results, start=1):
            if header:
                headers.append(header)
            total_incomplete += inc_count
            total_duplicates_local += dup_count
            # Добавляем строки
            all_rows.extend(rows)

        # Выберем "первый" заголовок как основной
        main_header = headers[0] if headers else None

        # Если включено remove_dup => нужно ещё убрать дубликаты МЕЖДУ файлами
        total_duplicates_global = 0
        if do_remove_dup:
            # глобальный set
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

        # Если слишком много строк
        if len(all_rows) > total_needed:
            all_rows = random.sample(all_rows, total_needed)

        # Перемешать, если нужно
        if do_shuffle:
            random.shuffle(all_rows)

        # Если заголовка нет, придумаем
        if not main_header and all_rows:
            col_count = len(all_rows[0])
            main_header = [f"col{i+1}" for i in range(col_count)]
        elif not main_header:
            # Совсем пусто
            main_header = []

        # Собираем DataFrame
        if not all_rows:
            msg = "В итоге не осталось ни одной строки (все удалены?)."
            self.after(0, lambda: messagebox.showinfo("Результат", msg, parent=self))
            return

        df = pd.DataFrame(all_rows, columns=main_header)

        # Если задана метка
        if label_value:
            df["label"] = label_value

        # Сохраняем
        try:
            df.to_csv(out_path, index=False)
        except Exception as e:
            msg = f"Не удалось сохранить итоговый файл: {e}"
            logger.error(msg)
            self.after(0, lambda: messagebox.showerror("Ошибка", msg, parent=self))
            return

        final_count = len(df)
        file_name = f"logs/{out_path.split('/')[-1].replace('.csv', '')}.log"
        msg = (
            f"Обработка завершена!\n\n"
            f"Файл сохранён: {out_path}\n"
            f"Всего файлов: {len(self.file_list)}\n\n"
            f"Некорректных строк (по файлам): {total_incomplete}\n"
            f"Дубликатов (внутри файлов): {total_duplicates_local}\n"
            f"Дубликатов (между файлами): {total_duplicates_global}\n"
            f"Итоговое кол-во строк: {final_count} (с учётом лимита = {total_needed})"
        )
        with open(file_name, 'w') as file:
            file.write(msg)

        # logger.info(msg)
        self.after(0, lambda: messagebox.showinfo("Результат", msg, parent=self))

    # ----------------- Прогресс по КОЛ-ВУ ФАЙЛОВ -----------------
    def set_progress(self, value, maximum):
        self.after(0, lambda: self._set_progress(value, maximum))

    def _set_progress(self, value, maximum):
        self.progress_bar["maximum"] = maximum
        self.progress_bar["value"] = value
        self.progress_label.config(text=f"Обработка файлов: {value}/{maximum}")

# --------------------------------------------------------------------------
#                         Главное окно
# --------------------------------------------------------------------------
class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Приложение: Скачивание и Нормализация CSV (Parallel)")
        self.configure(bg="#121212")

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#121212")
        style.configure("TLabel", background="#121212", foreground="cyan", font=("Courier", 12))
        style.configure("TButton", background="black", foreground="white", font=("Courier", 12, "bold"))
        style.map("TButton",
                  background=[("active", "#2d2d2d")],
                  foreground=[("active", "white")])

        # Получаем список папок (для скачивания)
        all_folders = [folder for folder in get_links(BASE_URL) if folder.endswith('/')]

        # Два "экрана"
        self.download_frame = DownloadFrame(self, all_folders)
        self.normalize_frame = NormalizeFrame(self)

        # Меню
        menubar = tk.Menu(self, background="black", foreground="white")
        self.config(menu=menubar)

        menu_functions = tk.Menu(menubar, tearoff=False, background="#2d2d2d", foreground="white")
        menu_functions.add_command(label="Скачать датасет", command=self.show_download)
        menu_functions.add_command(label="Нормализовать CSV (параллельно)", command=self.show_normalize)
        menubar.add_cascade(label="Меню", menu=menu_functions)

        menu_file = tk.Menu(menubar, tearoff=False, background="#2d2d2d", foreground="white")
        menu_file.add_command(label="Выход", command=self.quit)
        menubar.add_cascade(label="Файл", menu=menu_file)

        # По умолчанию показываем окно скачивания
        self.download_frame.pack(fill="both", expand=True)
        self.normalize_frame.pack_forget()

    def show_download(self):
        self.normalize_frame.pack_forget()
        self.download_frame.pack(fill="both", expand=True)

    def show_normalize(self):
        self.download_frame.pack_forget()
        self.normalize_frame.pack(fill="both", expand=True)

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
