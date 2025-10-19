
from dataclasses import dataclass, replace
from typing import List, Any, Callable
import pandas as pd
import streamlit as st
import plotly.express as px
from pycaret.regression import load_model, predict_model

# GŁÓWNE STRUKTURY DANYCH

@dataclass(frozen=True)
class AppConfig:
    """Przechowuje stałe konfiguracyjne aplikacji."""
    PAGE_TITLE: str = "Doradca Składki Ubezpieczeniowej AI"
    PAGE_ICON: str = "🩺"
    MODEL_PATH: str = 'fin'
    USD_TO_PLN_RATE: float = 4.0
    MONTHS_IN_YEAR: int = 12
    TARGET_BMI: float = 24.9
    MARKET_ADJUSTMENT_FACTOR: float = 0.2
    GROUP_POLICY_DISCOUNT: float = 0.85  # 15% zniżki
    ALCOHOL_UNITS_THRESHOLD: int = 7
    ACTIVITY_DAYS_THRESHOLD: int = 3

@dataclass(frozen=True)
class UserProfile:
    """Niezmienna struktura danych reprezentująca profil użytkownika."""
    age: int; sex: str; height_cm: float; weight_kg: float; smoker: bool
    children: int; weekly_activity_days: int; alcohol_units_week: int
    conditions: List[str]; region: str; has_group_option: bool
    prefers_higher_deductible: bool

    @property
    def bmi(self) -> float:
        height_m = max(self.height_cm / 100.0, 0.5)
        return round(self.weight_kg / (height_m ** 2), 1)

    def to_prediction_input(self) -> pd.DataFrame:
        """Konwertuje profil na format DataFrame wymagany przez model AI."""
        return pd.DataFrame({
            'age': [self.age], 'sex': [self.sex], 'bmi': [self.bmi],
            'children': [self.children], 'smoker': ['yes' if self.smoker else 'no'],
            'region': [self.region], 'weekly_activity_days': [self.weekly_activity_days],
            'alcohol_units_week': [self.alcohol_units_week],
            'has_conditions': [1 if self.conditions else 0]
        })

@dataclass(frozen=True)
class Recommendation:
    """Struktura dla pojedynczej rekomendacji i logiki jej symulacji."""
    id: str; title: str; description: str
    applies_when: Callable[[UserProfile], bool]
    simulate_change: Callable[[UserProfile], UserProfile]

@dataclass(frozen=True)
class AppState:
    """Agreguje cały stan aplikacji w jednym miejscu, ułatwiając przekazywanie danych."""
    profile: UserProfile
    pipeline: Any
    engine: Any  # RecommendationEngine
    config: AppConfig
    base_premium: float
    multiplier: int
    period_label: str

# MODEL/PREDYKCJE

@st.cache_resource
def load_pipeline(model_path: str) -> Any:
    """Wczytuje i cachuje wytrenowany model PyCaret."""
    # UWAGA: Usunęliśmy sprawdzanie istnienia pliku. Jeśli go brakuje, PyCaret zgłosi błąd.
    return load_model(model_path, verbose=False)

def _calculate_base_premium(u: UserProfile, pipeline: Any, config: AppConfig) -> float:
    """Oblicza bazową składkę przed nałożeniem zniżek produktowych."""
    input_df = u.to_prediction_input()
    pred_df = predict_model(pipeline, data=input_df)
    
    expected_usd_year = float(pred_df['prediction_label'].iloc[0])
    adjusted_usd_year = expected_usd_year * config.MARKET_ADJUSTMENT_FACTOR

    expected_loss_pln_year = adjusted_usd_year * config.USD_TO_PLN_RATE
    gross_premium_year = (expected_loss_pln_year / 0.75) * (0.85 if u.prefers_higher_deductible else 1.0)
    
    return gross_premium_year / config.MONTHS_IN_YEAR

def calculate_final_premium(u: UserProfile, pipeline: Any, config: AppConfig) -> float:
    """Oblicza finalną, miesięczną składkę dla użytkownika."""
    base_premium = _calculate_base_premium(u, pipeline, config)
    
    final_premium = base_premium
    if u.has_group_option:
        final_premium *= config.GROUP_POLICY_DISCOUNT
        
    return round(final_premium, 2)

class RecommendationEngine:
    """Zarządza logiką i biblioteką wszystkich dostępnych rekomendacji."""
    def __init__(self, config: AppConfig):
        self._config = config
        self._recommendations = self._initialize_recommendations()

    def _get_target_weight(self, height_cm: float) -> float:
        h_m = height_cm / 100.0
        return round(self._config.TARGET_BMI * (h_m ** 2), 1)

    def _initialize_recommendations(self) -> List[Recommendation]:
        """Definiuje centralną bibliotekę porad w systemie."""
        return [
            Recommendation("quit_smoking", "Rzuć palenie",
                "Największy pojedynczy czynnik ryzyka, przynoszący największe korzyści finansowe i zdrowotne.",
                lambda u: u.smoker, lambda u: replace(u, smoker=False)),
            Recommendation("improve_bmi", f"Zredukuj BMI do normy (< {self._config.TARGET_BMI})",
                "Osiągnięcie prawidłowej masy ciała znacznie obniża ryzyko wielu chorób przewlekłych.",
                lambda u: u.bmi >= self._config.TARGET_BMI, lambda u: replace(u, weight_kg=self._get_target_weight(u.height_cm))),
            Recommendation("increase_activity", "Zwiększ aktywność fizyczną",
                f"Regularna aktywność (min. {self._config.ACTIVITY_DAYS_THRESHOLD} dni w tyg.) jest kluczowa dla zdrowia.",
                lambda u: u.weekly_activity_days < self._config.ACTIVITY_DAYS_THRESHOLD, lambda u: replace(u, weekly_activity_days=self._config.ACTIVITY_DAYS_THRESHOLD)),
            Recommendation("reduce_alcohol", "Ogranicz spożycie alkoholu",
                f"Ograniczenie spożycia do max. {self._config.ALCOHOL_UNITS_THRESHOLD} jednostek tygodniowo poprawia profil ryzyka.",
                lambda u: u.alcohol_units_week > self._config.ALCOHOL_UNITS_THRESHOLD, lambda u: replace(u, alcohol_units_week=self._config.ALCOHOL_UNITS_THRESHOLD)),
            Recommendation("group_policy_benefit", "Zobacz korzyść z polisy grupowej",
                "Sprawdź, ile oszczędzasz dzięki tej opcji w porównaniu do standardowej oferty indywidualnej.",
                lambda u: u.has_group_option, lambda u: replace(u, has_group_option=False)),
        ]

    def get_for_user(self, user_profile: UserProfile) -> List[Recommendation]:
        """Zwraca listę rekomendacji pasujących do danego profilu."""
        return [r for r in self._recommendations if r.applies_when(user_profile)]


 # UI   


def ui_sidebar(config: AppConfig) -> UserProfile:
    """Tworzy panel boczny i zbiera dane od użytkownika, zwracając obiekt UserProfile."""
    st.sidebar.header("📝 Wprowadź swoje dane")
    with st.sidebar:
        age = st.number_input("Wiek", 18, 100, 30, key="age")
        sex_map = {"Kobieta": "female", "Mężczyzna": "male"}
        sex_display = st.selectbox("Płeć", list(sex_map.keys()), index=1, key="sex")
        height_cm = st.number_input("Wzrost [cm]", 120, 220, 180, key="height")
        weight_kg = st.number_input("Waga [kg]", 40, 250, 85, key="weight")
        st.divider()
        smoker = st.toggle("Czy palisz tytoń?", False, key="smoker")
        children = st.number_input("Liczba dzieci", 0, 10, 0, key="children")
        weekly_activity_days = st.slider("Dni z aktywnością fizyczną w tyg.", 0, 7, 1, key="activity")
        alcohol_units_week = st.slider("Jednostki alkoholu w tyg.", 0, 7, 5, key="alcohol")
        st.divider()
        conditions = st.multiselect("Choroby przewlekłe", ["nadciśnienie", "cukrzyca"], key="conditions")
        
        region_map = {
            "Zachodniopomorskie": "northwest", "Pomorskie": "northwest", "Kujawsko-Pomorskie": "northwest",
            "Wielkopolskie": "northwest", "Lubuskie": "northwest", "Warmińsko-Mazurskie": "northeast",
            "Podlaskie": "northeast", "Mazowieckie": "northeast", "Dolnośląskie": "southwest",
            "Opolskie": "southwest", "Śląskie": "southwest", "Łódzkie": "southeast", "Świętokrzyskie": "southeast",
            "Lubelskie": "southeast", "Podkarpackie": "southeast", "Małopolskie": "southeast"
        }
        region_display = st.selectbox("Województwo", list(region_map.keys()), index=1, key="region")
        st.divider()
        has_group_option = st.toggle("Masz opcję polisy grupowej?", True, key="group_option",
            help="Polisa oferowana przez pracodawcę, zazwyczaj na korzystniejszych warunkach.")
        prefers_higher_deductible = st.toggle("Rozważasz wyższy udział własny?", False, key="deductible",
            help="Oznacza niżsżą składkę w zamian za wzięcie na siebie większej części kosztów ewentualnej szkody.")

        return UserProfile(
            age=age, sex=sex_map[sex_display], height_cm=height_cm, weight_kg=weight_kg, smoker=smoker,
            children=children, weekly_activity_days=weekly_activity_days, alcohol_units_week=alcohol_units_week,
            conditions=conditions, region=region_map[region_display], has_group_option=has_group_option,
            prefers_higher_deductible=prefers_higher_deductible
        )

def ui_dashboard(state: AppState):
    """Wyświetla główny panel wskaźników (KPI)."""
    st.subheader("📊 Twoja spersonalizowana analiza")
    k1, k2, k3 = st.columns(3)
    bmi_status = "Poza normą" if state.profile.bmi >= 25 or state.profile.bmi < 18.5 else "Norma"
    k1.metric("Twoje BMI", f"{state.profile.bmi}", f"Status: {bmi_status}", delta_color="off")
    k2.metric("Szacunkowa składka", f"{state.base_premium * state.multiplier:.2f} zł{state.period_label}")
    k3.metric("Status palenia", "Palący 🚬" if state.profile.smoker else "Niepalący ✅")

def ui_recommendations(state: AppState):
    """Wyświetla sekcję z interaktywnymi rekomendacjami."""
    st.subheader("💡 Jak możesz realnie obniżyć składkę?")
    st.caption("Kliknij przycisk, aby zobaczyć precyzyjną symulację oszczędności.")

    active_recos = state.engine.get_for_user(state.profile)
    if not active_recos:
        st.success("Gratulacje! Twój profil jest bardzo dobry i nie mamy oczywistych rekomendacji.")
        return

    for reco in active_recos:
        with st.expander(f"**{reco.title}**"):
            st.write(reco.description)
            if st.button(f"Symuluj dla: {reco.title}", key=f"btn_{reco.id}"):
                modified_profile = reco.simulate_change(state.profile)
                new_premium = calculate_final_premium(modified_profile, state.pipeline, state.config)
                savings = (state.base_premium - new_premium) if reco.id != "group_policy_benefit" else (new_premium - state.base_premium)
                st.session_state.simulations[reco.id] = {"new_premium": new_premium, "savings": savings}
            
            if reco.id in st.session_state.simulations:
                sim = st.session_state.simulations[reco.id]
                sav = sim['savings'] * state.multiplier
                if sav > 0:
                    msg = (f"Oszczędność dzięki tej opcji: **{sav:.2f} zł{state.period_label}**"
                           if reco.id == "group_policy_benefit" else
                           f"Nowa składka: **{sim['new_premium'] * state.multiplier:.2f} zł{state.period_label}** | Oszczędność: **{sav:.2f} zł{state.period_label}**")
                    st.success(f"✅ {msg}")
                else:
                    st.warning("Ta zmiana w obecnym profilu nie przynosi dodatkowych oszczędności.")

def ui_savings_chart(state: AppState):
    """Wyświetla wykres słupkowy z podsumowaniem możliwych oszczędności."""
    if not st.session_state.get('simulations'):
        return
        
    st.subheader("Wizualizacja potencjalnych oszczędności")
    
    all_recos_map = {r.id: r.title for r in state.engine._recommendations}
    sim_data = [
        {"Rekomendacja": all_recos_map[reco_id], "Oszczędność": sim_data['savings'] * state.multiplier}
        for reco_id, sim_data in st.session_state.simulations.items() if sim_data['savings'] > 0
    ]

    if sim_data:
        df_sim = pd.DataFrame(sim_data).sort_values("Oszczędność", ascending=False)
        fig = px.bar(df_sim, x="Rekomendacja", y="Oszczędność",
                     title="Twoje realne oszczędności na podstawie symulacji",
                     labels={"Oszczędność": f"Oszczędność [zł{state.period_label}]", "Rekomendacja": ""},
                     text_auto='.2f')
        st.plotly_chart(fig, use_container_width=True)

# MAIN


def manage_session_state(current_profile: UserProfile):
    """Inicjalizuje lub resetuje stan symulacji, jeśli profil użytkownika się zmienił."""
    if 'simulations' not in st.session_state or st.session_state.get('last_profile') != current_profile:
        st.session_state.simulations = {}
        st.session_state.last_profile = current_profile

def main():
    """Główna funkcja uruchamiająca aplikację Streamlit."""
    config = AppConfig()
    st.set_page_config(page_title=config.PAGE_TITLE, page_icon=config.PAGE_ICON, layout="wide")
    st.title(f"{config.PAGE_ICON} {config.PAGE_TITLE}")
    
    pipeline = load_pipeline(config.MODEL_PATH)
    reco_engine = RecommendationEngine(config)
    user_profile = ui_sidebar(config)

    manage_session_state(user_profile)

    base_premium = calculate_final_premium(user_profile, pipeline, config)
    
    view = st.radio("Pokaż koszty:", ["Miesięcznie", "Rocznie"], horizontal=True, index=0)
    multiplier = config.MONTHS_IN_YEAR if view == "Rocznie" else 1
    period_label = "/rok" if view == "Rocznie" else "/mies"
    
    app_state = AppState(
        profile=user_profile, pipeline=pipeline, engine=reco_engine, config=config,
        base_premium=base_premium, multiplier=multiplier, period_label=period_label
    )
    
    st.divider()
    ui_dashboard(app_state)
    st.divider()
    ui_recommendations(app_state)
    st.divider()
    ui_savings_chart(app_state)

if __name__ == "__main__":
    main()