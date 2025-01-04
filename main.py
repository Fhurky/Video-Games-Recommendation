import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from collections import defaultdict
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import tkinter.font as tkFont
import threading
from PIL import Image, ImageTk
import webbrowser

class GameRecommenderApp:
    def __init__(self, root):
        # Veri yükleme ve hazırlama
        self.data = pd.read_csv("datasets/new_games_steam.csv")
        self.data = self.data.dropna(subset=['name'])
        self.data['name'] = self.data['name'].astype(str)
        
        # Model için veriyi hazırlama
        self.X = self.data.drop(["name"], axis=1)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Modelleri yükleme
        self.knn = joblib.load("knn_model.pkl")
        self.kmeans = joblib.load("kmeans_model.pkl")
        self.gmm = joblib.load("gmm_model.pkl")
        
        # Ana pencere ayarları
        self.root = root
        self.root.title("Akıllı Oyun Öneri Sistemi")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        # Loading animasyonu için değişkenler
        self.loading_frames = []
        self.current_frame = 0
        self.load_animation()
        
        # Tüm oyun adlarını al
        self.all_games = sorted(self.data['name'].unique())
        
        # Varsayılan ağırlıklar
        self.default_weights = {
            "knn": 3.2,
            "kmeans": 2.7,
            "gmm": 2.3
        }
        
        self.setup_gui()
    
    def load_animation(self):
        """Loading GIF'ini yükle ve boyutlandır"""
        loading_gif = Image.open("loading.gif")
        try:
            while True:
                frame = loading_gif.copy()
                frame = frame.resize((50, 50), Image.Resampling.LANCZOS)
                self.loading_frames.append(ImageTk.PhotoImage(frame))
                loading_gif.seek(loading_gif.tell() + 1)
        except EOFError:
            pass
    
    def setup_gui(self):
        """GUI bileşenlerini oluştur"""
        # Ana başlık
        title_font = tkFont.Font(family="Helvetica", size=16, weight="bold")
        title = Label(self.root, text="Oyun Öneri Sistemi", font=title_font, bg='#f0f0f0')
        title.pack(pady=20)
        
        self.create_search_section()
        self.create_weights_section()
        self.create_button_section()
        self.create_results_section()
    
    def create_search_section(self):
        """Arama bölümünü oluştur"""
        input_frame = ttk.LabelFrame(self.root, text="Oyun Arama", padding="10")
        input_frame.pack(fill="x", padx=20, pady=10)
        
        search_frame = ttk.Frame(input_frame)
        search_frame.pack(fill="x", expand=True)
        
        Label(search_frame, text="Oyun Adı:").pack(side=LEFT, padx=5)
        
        self.search_var = StringVar()
        self.search_var.trace('w', self.on_search_change)
        
        self.game_entry = ttk.Entry(search_frame, width=40, textvariable=self.search_var)
        self.game_entry.pack(side=LEFT, padx=5)
        
        # Öneri listesi
        self.suggestion_frame = ttk.Frame(input_frame)
        self.suggestion_list = Listbox(self.suggestion_frame, height=5)
        self.suggestion_list.pack(fill="x", expand=True)
        self.suggestion_list.bind('<<ListboxSelect>>', self.on_select)
    
    def create_weights_section(self):
        """Ağırlık ayarları bölümünü oluştur"""
        weights_frame = ttk.LabelFrame(self.root, text="Model Ağırlıkları", padding="10")
        weights_frame.pack(fill="x", padx=20, pady=10)
        
        self.use_custom_weights = BooleanVar(value=False)
        ttk.Checkbutton(weights_frame, text="Özel ağırlıklar kullan", 
                       variable=self.use_custom_weights, 
                       command=self.toggle_weight_entries).pack(pady=5)
        
        self.weight_entries = {}
        for model, default_weight in self.default_weights.items():
            frame = ttk.Frame(weights_frame)
            frame.pack(fill="x", pady=2)
            
            Label(frame, text=f"{model.upper()} ağırlığı:").pack(side=LEFT, padx=5)
            entry = ttk.Entry(frame, width=10, state='disabled')
            entry.insert(0, str(default_weight))
            entry.pack(side=LEFT, padx=5)
            self.weight_entries[model] = entry
    
    def create_button_section(self):
        """Buton bölümünü oluştur"""
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        
        style = ttk.Style()
        style.configure('Custom.TButton', padding=10)
        
        self.recommend_button = ttk.Button(button_frame, 
                                         text="Benzer Oyunları Bul",
                                         style='Custom.TButton',
                                         command=self.start_recommendation)
        self.recommend_button.pack(side=LEFT, padx=5)
        
        self.loading_label = Label(button_frame, image=self.loading_frames[0])
    
    def create_results_section(self):
        """Sonuç listesi ve arama butonu bölümünü oluştur"""
        result_frame = ttk.LabelFrame(self.root, text="Önerilen Oyunlar", padding="10")
        result_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Treeview oluşturma
        self.result_list = ttk.Treeview(result_frame, columns=("game", "score"), show="headings")
        self.result_list.heading("game", text="Oyun Adı")
        self.result_list.heading("score", text="Benzerlik Skoru")
        self.result_list.column("game", width=400)
        self.result_list.column("score", width=100)
        
        # Scrollbar ekleme
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_list.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_list.configure(yscrollcommand=scrollbar.set)
        self.result_list.pack(fill="both", expand=True)
        
        # Arama butonu için frame
        button_frame = ttk.Frame(result_frame)
        button_frame.pack(pady=10)
        
        # Google'da arama butonu
        self.search_button = ttk.Button(
            button_frame,
            text="Seçili Oyunu Google'da Ara",
            command=self.search_game_on_google,
            state='disabled'  # Başlangıçta devre dışı
        )
        self.search_button.pack()
        
        # Seçim yapıldığında butonu aktif hale getir
        self.result_list.bind('<<TreeviewSelect>>', self.on_game_select)
    
    def update_loading_animation(self):
        """Loading animasyonunu güncelle"""
        if self.loading_label and self.loading_label.winfo_exists():
            self.current_frame = (self.current_frame + 1) % len(self.loading_frames)
            self.loading_label.configure(image=self.loading_frames[self.current_frame])
            self.root.after(50, self.update_loading_animation)
    
    def start_recommendation(self):
        """Öneri işlemini başlat"""
        self.recommend_button.configure(state='disabled')
        self.loading_label.pack(side=LEFT, padx=5)
        self.current_frame = 0
        self.update_loading_animation()
        
        thread = threading.Thread(target=self.process_recommendation)
        thread.daemon = True
        thread.start()
    
    def process_recommendation(self):
        """Öneri işlemini gerçekleştir"""
        try:
            self.find_similar_games()
        finally:
            self.root.after(0, self.finish_recommendation)
    
    def finish_recommendation(self):
        """Öneri işlemini bitir"""
        self.loading_label.pack_forget()
        self.recommend_button.configure(state='normal')
    
    def on_search_change(self, *args):
        """Arama değiştiğinde tetiklenir"""
        search_term = self.search_var.get().lower()
        
        if search_term == "":
            self.suggestion_frame.pack_forget()
            return
        
        matching_games = [game for game in self.all_games 
                         if search_term in game.lower()][:10]
        
        self.suggestion_list.delete(0, END)
        for game in matching_games:
            self.suggestion_list.insert(END, game)
        
        if matching_games:
            self.suggestion_frame.pack(fill="x", expand=True)
        else:
            self.suggestion_frame.pack_forget()
    
    def on_select(self, event):
        """Öneri listesinden seçim yapıldığında tetiklenir"""
        if self.suggestion_list.curselection():
            selected_game = self.suggestion_list.get(self.suggestion_list.curselection())
            self.search_var.set(selected_game)
            self.suggestion_frame.pack_forget()
    
    def toggle_weight_entries(self):
        """Ağırlık girişlerini etkinleştir/devre dışı bırak"""
        state = 'normal' if self.use_custom_weights.get() else 'disabled'
        for entry in self.weight_entries.values():
            entry.configure(state=state)
    
    def get_current_weights(self):
        """Güncel ağırlıkları al"""
        if not self.use_custom_weights.get():
            return self.default_weights
        
        weights = {}
        try:
            for model, entry in self.weight_entries.items():
                weights[model] = float(entry.get())
            return weights
        except ValueError:
            messagebox.showerror("Hata", "Lütfen geçerli sayısal ağırlıklar giriniz.")
            return None
    
    def on_game_select(self, event):
        """Oyun seçildiğinde arama butonunu aktif hale getir"""
        selection = self.result_list.selection()
        if selection:
            self.search_button.configure(state='normal')
        else:
            self.search_button.configure(state='disabled')
    
    def search_game_on_google(self):
        """Seçili oyunu Google'da arat"""
        selection = self.result_list.selection()
        if selection:
            selected_item = self.result_list.item(selection[0])
            game_name = selected_item['values'][0]
            search_url = f"https://www.google.com/search?q={game_name}+game"
            webbrowser.open_new_tab(search_url)
    
    def find_similar_games(self):
        """Benzer oyunları bul"""
        input_game_name = self.game_entry.get()
        
        if input_game_name not in self.data['name'].values:
            messagebox.showerror("Hata", "Girilen oyun adı veri setinde bulunamadı.")
            return
        
        weights = self.get_current_weights()
        if weights is None:
            return
        
        target_game_index = self.data[self.data['name'] == input_game_name].index[0]
        target_game = input_game_name
        
        # Model tahminleri
        distances, indices = self.knn.kneighbors([self.X_scaled[target_game_index]])
        self.data['kmeans_cluster'] = self.kmeans.predict(self.X_scaled)
        self.data['gmm_cluster'] = self.gmm.predict(self.X_scaled)
        
        target_kmeans_cluster = self.data.loc[target_game_index, 'kmeans_cluster']
        target_gmm_cluster = self.data.loc[target_game_index, 'gmm_cluster']
        
        similar_games_kmeans = self.data[self.data['kmeans_cluster'] == target_kmeans_cluster]
        similar_games_gmm = self.data[self.data['gmm_cluster'] == target_gmm_cluster]
        
        # Sonuçları birleştir
        weighted_games = defaultdict(int)
        
        # Her modelden gelen sonuçları ağırlıklandır
        for i in indices[0]:
            game_name = self.data['name'].iloc[i]
            if game_name != target_game:
                weighted_games[game_name] += weights["knn"]
        
        for name in similar_games_kmeans['name'][:10]:
            if name != target_game:
                weighted_games[name] += weights["kmeans"]
        
        for name in similar_games_gmm['name'][:10]:
            if name != target_game:
                weighted_games[name] += weights["gmm"]
        
        self.root.after(0, self.update_results, weighted_games)
    
    def update_results(self, weighted_games):
        """Sonuçları göster"""
        self.result_list.delete(*self.result_list.get_children())
        sorted_games = sorted(weighted_games.items(), key=lambda x: x[1], reverse=True)
        
        for game, weight in sorted_games[:20]:
            self.result_list.insert("", "end", values=(game, f"{weight:.2f}"))

if __name__ == "__main__":
    root = Tk()
    app = GameRecommenderApp(root)
    root.mainloop()