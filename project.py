import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox, filedialog
from scipy.optimize import minimize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import random

# Розширений список акцій та криптовалют
stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'BABA', 'ORCL']
cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
assets = stocks + cryptos

import yfinance as yf
import pandas as pd
from datetime import datetime
from tkinter import messagebox

# Список активів
assets = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA']

# Функція для завантаження даних
def load_data():
    print("Data loading...")
    try:
        data = yf.download(assets, start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'))
        close_data = data['Close']  # Отримуємо тільки ціни закриття
        returns = close_data.pct_change().dropna()  # Рахуємо доходність

        return close_data, returns, returns.mean(), returns.cov()
    except Exception as e:
        messagebox.showerror("Помилка", f"Помилка завантаження даних: {e}")
        return None, None, None, None

# Викликаємо функцію для завантаження даних
close_data, returns, mean_returns, cov_returns = load_data()

# Перевірка результатів
if close_data is not None:
    print("Дані успішно завантажено!")
    print(close_data.head())
    print(returns.head())
else:
    print("Не вдалося завантажити дані.")


# Функції для розрахунку метрик портфеля
def portfolio_metrics(weights, mean_returns, cov_matrix, dividends=0, tax_rate=0):
    port_return = np.dot(weights, mean_returns) + dividends  # Врахування дивідендів
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = port_return / port_risk if port_risk != 0 else 0
    # Врахування податку на прибуток
    port_return_after_tax = port_return * (1 - tax_rate)
    return port_return_after_tax, port_risk, sharpe_ratio

# Алгоритм Марковица для оптимізації портфеля
def markowitz_optimization(mean_returns, cov_matrix, target_return=None):
    num_assets = len(mean_returns)
    
    def objective(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if target_return:
            return portfolio_risk
        return -portfolio_return / portfolio_risk

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_weights = np.array([1. / num_assets] * num_assets)
    
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
    return result.x if result.success else None

# Генетичний алгоритм для оптимізації портфеля
def genetic_algorithm_optimization(mean_returns, cov_matrix, population_size=100, generations=500, mutation_rate=0.1):
    num_assets = len(mean_returns)

    # Ініціалізація популяції
    population = np.random.rand(population_size, num_assets)
    population = population / population.sum(axis=1)[:, None]  # Нормалізація, щоб сума ваг була 1

    # Функція для оцінки фітнесу
    def fitness(weights):
        port_return, port_risk, _ = portfolio_metrics(weights, mean_returns, cov_matrix)
        return port_return / port_risk  # Максимізація коефіцієнта Шарпа

    # Генетичний алгоритм
    for generation in range(generations):
        # Оцінка фітнесу кожної особини
        fitness_values = np.array([fitness(individual) for individual in population])

        # Вибір двох кращих особин (турнірний відбір)
        sorted_indices = np.argsort(fitness_values)[-2:]
        parent1, parent2 = population[sorted_indices[-2]], population[sorted_indices[-1]]

        # Кросовер
        crossover_point = random.randint(1, num_assets - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

        # Мутація
        if random.random() < mutation_rate:
            mutation_index = random.randint(0, num_assets - 1)
            child1[mutation_index] = random.random()
        if random.random() < mutation_rate:
            mutation_index = random.randint(0, num_assets - 1)
            child2[mutation_index] = random.random()

        # Нормалізація
        child1 /= child1.sum()
        child2 /= child2.sum()

        # Заміна двох найгірших особин на нових дітей
        population[sorted_indices[0]] = child1
        population[sorted_indices[1]] = child2

    # Повернення найкращої особини
    best_individual = population[np.argmax(fitness_values)]
    return best_individual
    

# Створення ефективного фронту
def efficient_frontier(mean_returns, cov_matrix, num_portfolios=1000):
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_risk, _ = portfolio_metrics(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_risk
        results[2, i] = portfolio_return / portfolio_risk
    return results

# Інтерфейс
class InvestmentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Оптимізація портфеля")
        self.geometry("1000x800")

        # Вкладки
        self.tab_control = ttk.Notebook(self)
        self.optimization_tab = tk.Frame(self.tab_control)
        self.strategy_tab = tk.Frame(self.tab_control)
        self.graph_tab = tk.Frame(self.tab_control)

        self.tab_control.add(self.optimization_tab, text="Оптимізація")
        self.tab_control.add(self.strategy_tab, text="Стратегія")
        self.tab_control.add(self.graph_tab, text="Графіки")

        self.tab_control.pack(expand=1, fill="both")
        # Кнопки для кожного графіку на вкладці "Графіки"
        # Кнопки для графіків
        self.plot_efficient_frontier_button = tk.Button(self.graph_tab, text="Графік ефективного фронту", command=self.plot_efficient_frontier)
        self.plot_efficient_frontier_button.pack(padx=10, pady=10)

        self.plot_portfolio_button = tk.Button(self.graph_tab, text="Графік портфеля", command=self.plot_portfolio)
        self.plot_portfolio_button.pack(padx=10, pady=10)

        self.plot_optimization_results_button = tk.Button(self.graph_tab, text="Результати оптимізації", command=self.plot_optimization_results)
        self.plot_optimization_results_button.pack(padx=10, pady=10)

        # Змінна для збереження поточного графіка
        self.current_canvas = None

        #Вкладка стратегії
        
        self.strategy_label = tk.Label(self.strategy_tab, text="Оберіть стратегію інвестування:")
        self.strategy_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.strategy_var = tk.StringVar(value="conservative")
        self.conservative_button = tk.Radiobutton(self.strategy_tab, text="Консервативна", variable=self.strategy_var, value="conservative")
        self.conservative_button.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.moderate_button = tk.Radiobutton(self.strategy_tab, text="Помірна", variable=self.strategy_var, value="moderate")
        self.moderate_button.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.aggressive_button = tk.Radiobutton(self.strategy_tab, text="Агресивна", variable=self.strategy_var, value="aggressive")
        self.aggressive_button.grid(row=3, column=0, padx=10, pady=5, sticky="w")
         # Введення суми інвестицій для стратегії
        self.amount_label = tk.Label(self.strategy_tab, text="Введіть суму інвестицій:")
        self.amount_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")

        self.amount_entry = tk.Entry(self.strategy_tab)
        self.amount_entry.grid(row=5, column=0, padx=10, pady=10)

        # Кнопка для розрахунку результатів стратегії
        self.calculate_button = tk.Button(self.strategy_tab, text="Розрахувати результат", command=self.calculate_strategy_results)
        self.calculate_button.grid(row=6, column=0, padx=10, pady=10)
        # Вкладка оптимізації
        self.asset_type_label = tk.Label(self.optimization_tab, text="Оберіть тип активів:")
        self.asset_type_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.asset_type_var = tk.StringVar(value="stocks")
        self.stocks_button = tk.Radiobutton(self.optimization_tab, text="Акції", variable=self.asset_type_var, value="stocks", command=self.update_asset_list)
        self.stocks_button.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.cryptos_button = tk.Radiobutton(self.optimization_tab, text="Криптовалюти", variable=self.asset_type_var, value="cryptos", command=self.update_asset_list)
        self.cryptos_button.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.amount_label = tk.Label(self.optimization_tab, text="Введіть суму інвестицій (в доларах):")
        self.amount_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")

        self.amount_entry = tk.Entry(self.optimization_tab)
        self.amount_entry.grid(row=5, column=0, padx=10, pady=10)

        self.assets_listbox = tk.Listbox(self.optimization_tab, height=10, selectmode=tk.MULTIPLE)
        self.assets_listbox.grid(row=8, column=0, padx=10, pady=10)

        self.update_asset_list()

        self.optimize_button = tk.Button(self.optimization_tab, text="Оптимізувати портфель", command=self.optimize_portfolio)
        self.optimize_button.grid(row=9, column=0, padx=10, pady=10)

        # Додавання активів
        self.new_asset_name_label = tk.Label(self.optimization_tab, text="Назва нового активу:")
        self.new_asset_name_label.grid(row=10, column=0, padx=10, pady=10, sticky="w")

        self.new_asset_name_entry = tk.Entry(self.optimization_tab)
        self.new_asset_name_entry.grid(row=11, column=0, padx=10, pady=10)

        self.add_asset_button = tk.Button(self.optimization_tab, text="Додати актив", command=self.add_asset)
        self.add_asset_button.grid(row=12, column=0, padx=10, pady=10)
     

    def update_asset_list(self):
        selected_type = self.asset_type_var.get()
        self.assets_listbox.delete(0, tk.END)
        if selected_type == "stocks":
            for asset in stocks:
                self.assets_listbox.insert(tk.END, asset)
        elif selected_type == "cryptos":
            for asset in cryptos:
                self.assets_listbox.insert(tk.END, asset)

    def add_asset(self):
        new_asset_name = self.new_asset_name_entry.get()
        if new_asset_name:
            asset_type = self.asset_type_var.get()
            if asset_type == "stocks":
                stocks.append(new_asset_name)
            elif asset_type == "cryptos":
                cryptos.append(new_asset_name)

            # Оновлення списку активів
            self.update_asset_list()

            # Очищення поля вводу для нового активу
            self.new_asset_name_entry.delete(0, tk.END)
            messagebox.showinfo("Успіх", f"Актив '{new_asset_name}' успішно додано.")
        else:
            messagebox.showerror("Помилка", "Будь ласка, введіть назву активу.")

    def optimize_portfolio(self):
        try:
            selected_assets = [self.assets_listbox.get(i) for i in self.assets_listbox.curselection()]
            print(f"Selected assets: {selected_assets}")
        
            if not selected_assets:
                raise ValueError("Оберіть активи для портфеля")

            amount_str = self.amount_entry.get()
            if not amount_str or not amount_str.isdigit():
                raise ValueError("Введіть правильну суму інвестицій")
        
            amount = float(amount_str)
            if amount <= 0:
                raise ValueError("Введіть правильну суму інвестицій")

            print(f"Available assets: {stocks + cryptos}")

            if not all(asset in stocks + cryptos for asset in selected_assets):
                raise ValueError("Оберіть тільки доступні активи")

            selected_mean_returns = mean_returns[selected_assets].values
            selected_cov_matrix = cov_returns.loc[selected_assets, selected_assets].values

            weights = markowitz_optimization(selected_mean_returns, selected_cov_matrix)
            if weights is None:
                raise ValueError("Неможливо оптимізувати портфель")

            port_return, port_risk, sharpe_ratio = portfolio_metrics(weights, selected_mean_returns, selected_cov_matrix)

            messagebox.showinfo("Результати оптимізації", f"Очікувана прибутковість: {port_return * 100:.2f}%\nРизик портфеля: {port_risk * 100:.2f}%\nШарп-коефіцієнт: {sharpe_ratio:.2f}")
    
        except ValueError as ve:
            messagebox.showerror("Помилка", str(ve))
        except Exception as e:
            messagebox.showerror("Помилка", f"Сталася помилка: {e}")
    
    #Вкладка для графіків 
    # Графік ефективного фронту
    def plot_efficient_frontier(self):
    # Видаляємо попередній графік, якщо він існує
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()

    # Отримуємо ефективний фронт
        results = efficient_frontier(mean_returns, cov_returns)

    # Створення графіку
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(results[1], results[0], c=results[2], cmap='viridis', marker='o')
        ax.set_xlabel("Ризик (Стандартне відхилення)")
        ax.set_ylabel("Доходність")
        ax.set_title("Ефективний фронт портфеля")

    # Додавання лінії найкращого портфеля
        optimal_weights = markowitz_optimization(mean_returns, cov_returns)
        if optimal_weights is not None:
            optimal_return, optimal_risk, _ = portfolio_metrics(optimal_weights, mean_returns, cov_returns)
            ax.scatter(optimal_risk, optimal_return, color='red', marker='*', s=200, label="Оптимальний портфель")

        ax.legend()

    # Відображення графіку в Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.graph_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Оновлюємо поточний canvas
        self.current_canvas = canvas


    # Графік портфеля
    def plot_portfolio(self):
    # Видаляємо попередній графік, якщо він існує
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()

    # Різні варіанти ваг активів
        num_portfolios = 10000
        results = np.zeros((3, num_portfolios))
        for i in range(num_portfolios):
            weights = np.random.random(len(mean_returns))
            weights /= np.sum(weights)  # Нормалізуємо ваги
            portfolio_return, portfolio_risk, _ = portfolio_metrics(weights, mean_returns, cov_returns)
            results[0,i] = portfolio_return
            results[1,i] = portfolio_risk
            results[2,i] = portfolio_return / portfolio_risk  # Коефіцієнт Шарпа

    # Створення графіку
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(results[1], results[0], c=results[2], cmap='viridis', marker='o')
        ax.set_xlabel("Ризик (Стандартне відхилення)")
        ax.set_ylabel("Доходність")
        ax.set_title("Різні портфелі")

    # Відображення графіку в Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.graph_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Оновлюємо поточний canvas
        self.current_canvas = canvas

    def plot_optimization_results(self):
    # Видаляємо попередній графік, якщо він існує
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()

    # Виконуємо оптимізацію портфеля
        optimal_weights = markowitz_optimization(mean_returns, cov_returns)

    # Розраховуємо доходність і ризик для оптимального портфеля
        if optimal_weights is not None:
            optimal_return, optimal_risk, _ = portfolio_metrics(optimal_weights, mean_returns, cov_returns)

        # Створення графіку
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(optimal_risk, optimal_return, color='red', marker='*', s=200, label="Оптимальний портфель")
            ax.set_xlabel("Ризик (Стандартне відхилення)")
            ax.set_ylabel("Доходність")
            ax.set_title("Результати оптимізації портфеля")
            ax.legend()

        # Відображення графіку в Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.graph_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Оновлюємо поточний canvas
            self.current_canvas = canvas    
    def calculate_strategy_results(self):
        selected_strategy = self.strategy_var.get()
        amount_str = self.amount_entry.get()

        try:
            amount = float(amount_str)
            if amount <= 0:
                raise ValueError("Сума інвестицій повинна бути більше 0.")

            # Залежно від стратегії, розраховуємо результат
            if selected_strategy == "conservative":
                result = self.simulate_conservative_strategy(amount)
            elif selected_strategy == "moderate":
                result = self.simulate_moderate_strategy(amount)
            elif selected_strategy == "aggressive":
                result = self.simulate_aggressive_strategy(amount)

            # Показуємо результат
            messagebox.showinfo("Результат стратегії", result)

        except ValueError as ve:
            messagebox.showerror("Помилка", str(ve))

    def simulate_conservative_strategy(self, amount):
        # Імітація результату для консервативної стратегії
        expected_return = 0.05  # 5% прибутку
        risk = 0.02  # 2% ризик
        result = amount * (1 + expected_return)
        return f"Консервативна стратегія: інвестиція {amount} дасть {result} через рік з ризиком {risk * 100}%."

    def simulate_moderate_strategy(self, amount):
        # Імітація результату для помірної стратегії
        expected_return = 0.1  # 10% прибутку
        risk = 0.1  # 10% ризик
        result = amount * (1 + expected_return)
        return f"Помірна стратегія: інвестиція {amount} дасть {result} через рік з ризиком {risk * 100}%."

    def simulate_aggressive_strategy(self, amount):
        # Імітація результату для агресивної стратегії
        expected_return = 0.2  # 20% прибутку
        risk = 0.3  # 30% ризик
        result = amount * (1 + expected_return)
        return f"Агресивна стратегія: інвестиція {amount} дасть {result} через рік з ризиком {risk * 100}%. "
    
if __name__ == "__main__":
    app = InvestmentApp()
    app.mainloop()
