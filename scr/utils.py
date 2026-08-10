import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def review_data(df: pd.DataFrame, max_unique: int = 10, show_value_counts: bool = True) -> dict:
    """
    Быстрый обзор датасета.
    
    df: DataFrame для анализа
    max_unique: Сколько уникальных значений показывать
    show_value_counts: Для категориальных — показывать value_counts вместо списка unique
    
    Возвращает словарь с основными метриками
    """
    print(f"📂 Размерность: {df.shape[0]:,} строк × {df.shape[1]} столбцов\n")
    
    # 1. Пропуски
    missing = df.isna().sum()
    missing_percent = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Пропуски': missing,
        'Пропуски (%)': missing_percent
    })
    
    total_missing = missing.sum()
    if total_missing > 0:
        print(f"📉 Пропуски: {total_missing:,} ({missing_percent.mean():.2f}% в среднем)")
        print(missing_df[missing_df['Пропуски'] > 0].sort_values('Пропуски (%)', ascending=False))
    else:
        print("✅ Пропусков нет")
    
    # 2. Дубликаты
    duplicates = df.duplicated().sum()
    print(f"\n🔄 Явные дубликаты: {duplicates:,}")
    
    # 3. Общая информация по колонкам
    print("\n🔍 Типы данных и уникальные значения:")
    summary = []
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        n_unique = df[col].nunique()
        missing_pct = missing_percent[col]
        
        print(f"• {col:25} | тип: {col_type:8} | уник: {n_unique:5,} | пропуски: {missing_pct:5.1f}%")
        
        # Дополнительный вывод для категориальных/малых кардинальностей
        if show_value_counts and (df[col].dtype == 'object' or n_unique <= max_unique):
            top_values = df[col].value_counts().head(max_unique)
            if not top_values.empty:
                print("     Топ значений:")
                for val, cnt in top_values.items():
                    print(f"       {val}: {cnt:,}")
        
        summary.append({
            'column': col,
            'dtype': col_type,
            'n_unique': n_unique,
            'missing_pct': missing_pct
        })
    
    print(f"\n{'='*80}")
    
    return {
        'shape': df.shape,
        'total_missing': total_missing,
        'duplicates': duplicates,
        'columns': summary
    }

def plot_distribution(df: pd.DataFrame, 
                      column: str, 
                      title: str = None,
                      figsize: tuple = (9, 3)):
    """
    Строит гистограмму, диаграмму размаха и выводит основные статистики
    """
    if column not in df.columns:
        print(f"Столбец {column} не найден!")
        return
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        print(f"Столбец {column} не числовой!")
        return
    
    if title is None:
        title = column
    
    # Статистика
    stats = df[column].describe()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Гистограмма
    bins = np.histogram_bin_edges(df[column].dropna(), bins='doane')
    sns.histplot(df[column], bins=bins, kde=True, ax=axes[0])
    axes[0].set_title(f'Гистограмма: {title}')
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Частота')
    
    # Диаграмма размаха
    sns.boxplot(x=df[column], ax=axes[1])
    axes[1].set_title(f'Диаграмма размаха: {title}')
    axes[1].set_xlabel(column)
    
    plt.tight_layout()
    plt.show()
    
    # Основные статистики
    print(f"\nСтатистика по {title}:")
    print(f"Среднее: {stats['mean']:.4f} | Медиана: {stats['50%']:.4f}")
    print(f"Минимум: {stats['min']:.4f} | Максимум: {stats['max']:.4f}")

def plot_matrix(df: pd.DataFrame, 
                cols: list = None, 
                target: str = None,
                title: str = "Матрица корреляций",
                figsize: tuple = (6, 5),
                annot: bool = True,
                method: str = 'pearson'):
    """
    Строит тепловую карту корреляций.
        df: DataFrame
        cols: Список колонок (если None — все числовые)
        target: Название целевой переменной (выделит её)
        method: 'pearson', 'spearman' или 'kendall'
    """
    if cols is None: # Только числовые колонки
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cols = numeric_cols
    
    corr_matrix = df[cols].corr(method=method)
    
    # Настраиваем размер в зависимости от количества признаков
    if figsize is None:
        size = max(8, len(cols) * 0.7)
        figsize = (size, size)
    
    plt.figure(figsize=figsize)
    
    # Маска для верхнего треугольника (чтобы не дублировать)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        annot=annot, 
        cmap='coolwarm', 
        fmt='.2f',
        mask=mask if len(cols) > 6 else None,  # маску включаем только на больших матрицах
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title(title, fontsize=14, pad=20)
    plt.xticks(ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_pairplot(df, height=1.6, diag_kind='hist', alpha=0.5, s=2, title=None, hue=None):
    g = sns.pairplot(
        df, corner=True, hue=hue,
        height=height, 
        diag_kind=diag_kind,
        plot_kws={'alpha': alpha, 's': s})

    g.fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()