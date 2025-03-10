import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def replace_negatives_and_infs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждого столбца DataFrame:
    1. Заменяет все отрицательные значения (включая -inf) на минимальное 
       положительное значение из этого столбца.
    2. Заменяет +inf на максимально возможное конечное значение из этого столбца.
    Если в столбце нет положительных значений, строит график этого столбца
    и выбрасывает исключение.
    """
    for col in df.columns:
        # Если столбец не числовой — пропускаем или меняем логику
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Столбец '{col}' не является числовым. Пропускаем...")
            continue

        # 1. Обработка отрицательных значений (включая -inf)
        positive_values = df.loc[(df[col] >= 0) & (~np.isinf(df[col])), col]

        if positive_values.empty:
            # Строим график значений столбца при отсутствии положительных значений
            plt.figure(figsize=(8, 4))
            plt.plot(df.index, df[col], marker='o', linestyle='-')
            plt.title(f"Нет положительных значений в столбце '{col}'")
            plt.xlabel("Номер строки")
            plt.ylabel(f"Значения столбца '{col}'")
            
            # Сохраняем график в файл (можно назвать как-то уникально)
            plot_filename = f"{col}_no_positive_values.png"
            plt.savefig(plot_filename)
            plt.close()
            
            # Выбрасываем исключение после сохранения графика
            print(f"Столбец '{col}' не содержит положительных значений. "
                  f"График сохранён в '{plot_filename}'.")
            raise ValueError(
                f"Столбец '{col}' не содержит положительных значений. "
                f"График сохранён в '{plot_filename}'."
            )

        min_positive = positive_values.min()
        # Заменяем все отрицательные на min_positive
        df.loc[df[col] < 0, col] = min_positive

        # 2. Обработка +inf
        if (df[col] == np.inf).any():
            max_finite = df.loc[~np.isinf(df[col]), col].max()
            df[col] = df[col].replace(np.inf, max_finite)

    return df


if __name__ == "__main__":
    # Пример использования
    file_path = "dataset_container/2023data/dataset1.csv"
    df = pd.read_csv(file_path)

    try:
        df = replace_negatives_and_infs(df)
    except ValueError as e:
        print("Ошибка при обработке столбцов:", e)
        # Если хотите продолжить обработку остальных столбцов — 
        # придётся менять логику, чтобы не прерывать весь цикл.

    print("\nРезультат обработки (describe):")
    print(df.describe())

    # Сохраняем результат обратно в CSV (если дошли до конца без ошибок)
    output_path = "dataset_container/2023data/dataset1.csv"
    df.to_csv(output_path, index=False)
    print(f"\nОбновлённый файл сохранён как '{output_path}'")
