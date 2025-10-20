from dataclasses import dataclass, replace
from typing import List, Any, Callable, Optional
import pandas as pd
import streamlit as st
import plotly.express as px
from pycaret.regression import load_model, predict_model

#  STRUKTURY DANYCH 

@dataclass(frozen=True)
class AppConfig:
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
    age: int; sex: str; height_cm: float; weight_kg: float; smoker: bool
    children: int; weekly_activity_days: int; alcohol_units_week: int
    conditions: List[str]; region: str; has_group_option: bool
    prefers_higher_deductible: bool

    @property
    def bmi(self) -> float:
        height_m = max(self.height_cm / 100.0, 0.5)
        return round(self.weight_kg / (height_m ** 2), 1)

    def to_prediction_input(self) -> pd.DataFrame:
        return pd.DataFrame({
            'age': [self.age], 'sex': [self.sex], 'bmi': [self.bmi],
            'children': [self.children], 'smoker': ['yes' if self.smoker else 'no'],
            'region': [self.region], 'weekly_activity_days': [self.weekly_activity_days],
            'alcohol_units_week': [self.alcohol_units_week],
            'has_conditions': [1 if self.conditions else 0]
        })

@dataclass(frozen=True)
class Recommendation:
    id: str; title: str; description: str
    health_impact: Optional[str]
    applies_when: Callable[[UserProfile], bool]
    simulate_change: Callable[[UserProfile], UserProfile]

@dataclass(frozen=True)
class AppState:
    """Główny stan aplikacji."""
    profile: UserProfile; pipeline: Any; engine: Any; config: AppConfig
    base_premium: float; multiplier: int; period_label: str
    
# MODEL  I PREDYKCJA

@st.cache_resource
def load_pipeline(model_path: str) -> Any:
    return load_model(model_path, verbose=False)

def _calculate_base_premium(u: UserProfile, pipeline: Any, config: AppConfig) -> float:
    input_df = u.to_prediction_input()
    pred_df = predict_model(pipeline, data=input_df)
    
    expected_usd_year = float(pred_df['prediction_label'].iloc[0])
    adjusted_usd_year = expected_usd_year * config.MARKET_ADJUSTMENT_FACTOR

    expected_loss_pln_year = adjusted_usd_year * config.USD_TO_PLN_RATE
    gross_premium_year = (expected_loss_pln_year / 0.75) * (0.85 if u.prefers_higher_deductible else 1.0)
    
    return gross_premium_year / config.MONTHS_IN_YEAR

def calculate_final_premium(u: UserProfile, pipeline: Any, config: AppConfig) -> float:
    base_premium = _calculate_base_premium(u, pipeline, config)
    final_premium = base_premium * config.GROUP_POLICY_DISCOUNT if u.has_group_option else base_premium
    return round(final_premium, 2)

class RecommendationEngine:

    def __init__(self, config: AppConfig):
        self._config = config
        self._recommendations = self._initialize_recommendations()

    def _get_target_weight(self, height_cm: float) -> float:
        h_m = height_cm / 100.0
        return round(self._config.TARGET_BMI * (h_m ** 2), 1)

    def _initialize_recommendations(self) -> List[Recommendation]:
        return [
            Recommendation(
                id="quit_smoking", title="Rzuć palenie",
                description="Największy pojedynczy czynnik ryzyka, przynoszący największe korzyści finansowe i zdrowotne.",
                health_impact="Palenie tytoniu drastycznie zwiększa ryzyko chorób serca, nowotworów (szczególnie płuc) i przewlekłych problemów z oddychaniem. Rzucenie palenia to najważniejszy krok w kierunku dłuższego życia.",
                applies_when=lambda u: u.smoker, 
                simulate_change=lambda u: replace(u, smoker=False)
            ),
            Recommendation(
                id="improve_bmi", title=f"Zredukuj BMI do normy (< {self._config.TARGET_BMI})",
                description="Osiągnięcie prawidłowej masy ciała znacznie obniża ryzyko wielu chorób przewlekłych, co przekłada się na składkę.",
                health_impact="Nadwaga i otyłość (BMI >= 25) to prosta droga do nadciśnienia, cukrzycy typu 2, chorób serca i problemów ze stawami. Utrzymanie prawidłowej wagi to fundament profilaktyki zdrowotnej.",
                applies_when=lambda u: u.bmi >= 25.0, 
                simulate_change=lambda u: replace(u, weight_kg=self._get_target_weight(u.height_cm))
            ),
            Recommendation(
                id="increase_activity", title="Zwiększ aktywność fizyczną",
                description=f"Zwiększenie aktywności do co najmniej {self._config.ACTIVITY_DAYS_THRESHOLD} dni w tygodniu to klucz do lepszego zdrowia i niższej składki.",
                health_impact=f"Niski poziom aktywności fizycznej jest jednym z głównych czynników ryzyka chorób cywilizacyjnych. Regularny ruch (nawet 30-minutowy spacer) pomaga regulować ciśnienie krwi, obniża poziom złego cholesterolu (LDL) i cukru we krwi, co bezpośrednio zmniejsza ryzyko zawału serca, udaru mózgu oraz cukrzycy typu 2. To inwestycja w dłużesze, zdrowsze życie.",
                applies_when=lambda u: u.weekly_activity_days < self._config.ACTIVITY_DAYS_THRESHOLD, 
                simulate_change=lambda u: replace(u, weekly_activity_days=self._config.ACTIVITY_DAYS_THRESHOLD)
            ),
            Recommendation(
                id="reduce_alcohol", title="Ogranicz spożycie alkoholu",
                description=f"Ograniczenie spożycia do maksymalnie {self._config.ALCOHOL_UNITS_THRESHOLD} jednostek tygodniowo poprawia profil ryzyka.",
                health_impact=f"Regularne spożywanie powyżej {self._config.ALCOHOL_UNITS_THRESHOLD} jednostek alkoholu tygodniowo znacząco obciąża wątrobę i zwiększa ryzyko jej marskości, a także chorób serca i niektórych nowotworów.",
                applies_when=lambda u: u.alcohol_units_week > self._config.ALCOHOL_UNITS_THRESHOLD, 
                simulate_change=lambda u: replace(u, alcohol_units_week=self._config.ALCOHOL_UNITS_THRESHOLD)
            ),
            Recommendation(
                id="group_policy_benefit", title="Zobacz korzyść z polisy grupowej",
                description="Sprawdź, ile oszczędzasz dzięki tej opcji w porównaniu do standardowej oferty indywidualnej.",
                health_impact=None, # Ta rekomendacja nie ma bezpośredniego wpływu na zdrowie
                applies_when=lambda u: u.has_group_option, 
                simulate_change=lambda u: replace(u, has_group_option=False)
            ),
        ]

    def get_for_user(self, user_profile: UserProfile) -> List[Recommendation]:
        """Zwraca pasujące rekomendacje finansowe."""
        return [r for r in self._recommendations if r.applies_when(user_profile)]

# UI

def ui_sidebar(config: AppConfig) -> UserProfile:
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
        has_group_option = st.toggle("Masz opcję polisy grupowej?", True, key="group_option", help="Polisa oferowana przez pracodawcę, zazwyczaj na korzystniejszych warunkach.")
        prefers_higher_deductible = st.toggle("Rozważasz wyższy udział własny?", False, key="deductible", help="Oznacza niżsżą składkę w zamian za wzięcie na siebie większej części kosztów ewentualnej szkody.")

        return UserProfile(
            age=age, sex=sex_map[sex_display], height_cm=height_cm, weight_kg=weight_kg, smoker=smoker,
            children=children, weekly_activity_days=weekly_activity_days, alcohol_units_week=alcohol_units_week,
            conditions=conditions, region=region_map[region_display], has_group_option=has_group_option,
            prefers_higher_deductible=prefers_higher_deductible
        )

def ui_dashboard(state: AppState):
    st.subheader("📊 Twoja spersonalizowana analiza")
    k1, k2, k3 = st.columns(3)

    with k1:
        bmi = state.profile.bmi
        color, status = ("green", "Prawidłowa ✅") if 18.5 <= bmi < 25 else ("red", "Poza normą ⚠️")
        st.markdown(f"""
        <div style="line-height: 1.2; height: 100%;"><p style="font-size: 0.9rem; color: #808495; margin-bottom: 0;">Twoje BMI</p><p style="font-size: 1.75rem; font-weight: 600; margin-bottom: 0;">{bmi}</p><p style="color: {color}; margin-bottom: 0;">{status}</p></div>
        """, unsafe_allow_html=True)

    k2.metric("Szacunkowa składka", f"{state.base_premium * state.multiplier:.2f} zł{state.period_label}")
    k3.metric("Status palenia", "Palący 🚬" if state.profile.smoker else "Niepalący ✅")

def ui_recommendations(state: AppState):
    st.subheader("💡 Jak możesz realnie obniżyć składkę i zadbać o zdrowie?")
    st.caption("Kliknij przycisk, aby zobaczyć precyzyjną symulację oszczędności.")

    active_recos = state.engine.get_for_user(state.profile)
    if not active_recos:
        st.success("Gratulacje! Twój profil jest bardzo dobry i nie mamy oczywistych rekomendacji.")
        return

    for reco in active_recos:
        with st.expander(f"**{reco.title}**"):
            st.write(reco.description)
            
            if reco.health_impact:
                st.info(f"**Wpływ na zdrowie:** {reco.health_impact}")

            if st.button(f"Symuluj dla: {reco.title}", key=f"btn_{reco.id}"):
                modified_profile = reco.simulate_change(state.profile)
                new_premium = calculate_final_premium(modified_profile, state.pipeline, state.config)
                savings = (state.base_premium - new_premium) if reco.id != "group_policy_benefit" else (new_premium - state.base_premium)
                st.session_state.simulations[reco.id] = {"new_premium": new_premium, "savings": savings}
            
            if reco.id in st.session_state.simulations:
                sim = st.session_state.simulations[reco.id]
                sav = sim['savings'] * state.multiplier
                if sav > 0.01:
                    msg = (f"Oszczędność dzięki tej opcji: **{sav:.2f} zł{state.period_label}**" if reco.id == "group_policy_benefit" else f"Nowa składka: **{sim['new_premium'] * state.multiplier:.2f} zł{state.period_label}** | Oszczędność: **{sav:.2f} zł{state.period_label}**")
                    st.success(f"✅ {msg}")
                else:
                    st.info("ℹ️ Ta symulacja nie pokazuje oszczędności dla Twojego obecnego profilu.")

def ui_savings_chart(state: AppState):
    """Renderuje wykres oszczędności."""
    if not st.session_state.get('simulations'):
        return
        
    st.subheader("Wizualizacja oszczędności")
    
    all_recos_map = {r.id: r.title for r in state.engine._recommendations}
    plot_data = []
    
    for reco_id, sim_data in st.session_state.simulations.items():
        if sim_data['savings'] > 0.01:
            reco_title = all_recos_map.get(reco_id, "Nieznana rekomendacja")
            base_premium_adj = state.base_premium * state.multiplier
            new_premium_adj = sim_data['new_premium'] * state.multiplier

            plot_data.extend([
                {'Rekomendacja': reco_title, 'Składka': base_premium_adj, 'Typ': 'Składka obecna'},
                {'Rekomendacja': reco_title, 'Składka': new_premium_adj, 'Typ': 'Składka po zmianie'}
            ])

    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        fig = px.bar(df_plot, x="Rekomendacja", y="Składka", color="Typ", barmode='group', title="Porównanie składek", labels={"Składka": f"Składka [zł{state.period_label}]", "Rekomendacja": "", "Typ": "Rodzaj składki"}, text_auto='.2f')
        fig.update_traces(textangle=0, textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

# MAIN  

def manage_session_state(current_profile: UserProfile):
    if 'simulations' not in st.session_state or st.session_state.get('last_profile') != current_profile:
        st.session_state.simulations = {}
        st.session_state.last_profile = current_profile

def main():
    """Główna funkcja aplikacji."""
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
        base_premium=base_premium, multiplier=multiplier, 
        period_label=period_label
    )
    
    st.divider()
    ui_dashboard(app_state)
    st.divider()
    ui_recommendations(app_state)
    st.divider()
    ui_savings_chart(app_state)

if __name__ == "__main__":
    main()
