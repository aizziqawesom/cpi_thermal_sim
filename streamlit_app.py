"""
app/streamlit_app.py
CPI Thermal Stress Simulator — Interactive Streamlit Demo
Run: streamlit run app/streamlit_app.py
"""

import sys
sys.path.append('..')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from src.cpi_model import (
    MATERIALS, PACKAGES, JEDEC_PROFILES,
    run_cpi_assessment, suhir_interface_stress,
    timoshenko_warpage, engelmaier_fatigue_life,
    thermal_stack_analysis, doe_sweep,
    Material, PackageGeometry
)

st.set_page_config(page_title="CPI Thermal Stress Simulator", page_icon="⚙️",
                   layout="wide", initial_sidebar_state="expanded")

# ── Sidebar: Inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Package Configuration")

    st.subheader("Die")
    die_size    = st.slider("Die size [mm]", 5.0, 25.0, 10.0, 0.5)
    die_thick   = st.slider("Die thickness [mm]", 0.1, 0.8, 0.3, 0.05)

    st.subheader("Substrate")
    sub_cte  = st.slider("Substrate CTE [ppm/°C]", 3.0, 21.0, 17.0, 0.5)
    sub_E    = st.slider("Substrate modulus [GPa]", 10.0, 50.0, 25.0, 1.0)
    sub_t    = st.slider("Substrate thickness [mm]", 0.4, 1.5, 0.8, 0.1)

    st.subheader("Underfill")
    uf_E    = st.slider("Underfill modulus [GPa]", 2.0, 14.0, 8.5, 0.5)
    uf_cte  = st.slider("Underfill CTE [ppm/°C]", 15.0, 35.0, 26.0, 1.0)

    st.subheader("Solder Bump")
    bump_pitch  = st.slider("Bump pitch [µm]", 50, 300, 150, 10)
    bump_height = st.slider("Bump height [µm]", 30, 150, 80, 5)

    st.subheader("Thermal Profile")
    profile_name = st.selectbox("JEDEC Profile", list(JEDEC_PROFILES.keys()), index=0)
    power_W      = st.slider("Die power dissipation [W]", 0.5, 15.0, 3.0, 0.5)

    st.markdown("---")
    st.caption("CPI Thermal Stress Simulator | Suhir + Engelmaier + Timoshenko")

# ── Build models from inputs ──────────────────────────────────────────────────
die = MATERIALS['silicon']
substrate = Material('Custom Sub', E=sub_E, nu=0.39, CTE=sub_cte, k=0.35, rho_Cp=1.5e6)
underfill = Material('Custom UF',  E=uf_E,  nu=0.35, CTE=uf_cte,  k=0.7,  rho_Cp=1.8e6)
geometry  = PackageGeometry(
    die_size=die_size, die_thickness=die_thick,
    substrate_size=die_size + 7, substrate_thickness=sub_t,
    bump_pitch=bump_pitch, bump_height=bump_height, bump_diameter=int(bump_pitch * 0.6),
    underfill_thickness=bump_height, mold_thickness=0.0
)
profile = JEDEC_PROFILES[profile_name]

result = run_cpi_assessment(die, substrate, underfill, geometry, profile, power_W=power_W)
s = result.summary()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚙️ CPI Thermal Stress Simulator")
st.caption(f"Flip-Chip BGA | {profile.name} ({profile.T_min}°C → {profile.T_max}°C) | Suhir + Engelmaier + Timoshenko Models")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
risk_color = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴', 'Critical': '🚨'}.get(s['risk'], '⚪')

col1.metric("Max Shear Stress τ", f"{s['tau_max_MPa']:.1f} MPa",
            delta=f"{'RISK' if s['tau_max_MPa'] > 50 else 'OK'}", delta_color="inverse" if s['tau_max_MPa'] > 50 else "normal")
col2.metric("Peel Stress σ_peel", f"{s['sigma_peel_MPa']:.1f} MPa")
col3.metric("Von Mises σ_vm", f"{s['sigma_vm_MPa']:.1f} MPa")
col4.metric("Warpage", f"{s['warpage_um']:.0f} µm",
            delta=f"{'JEDEC FAIL' if not s['passes_jedec_warpage'] else 'JEDEC PASS'}",
            delta_color="inverse" if not s['passes_jedec_warpage'] else "normal")
col5.metric(f"Fatigue Life N_f {risk_color}", f"{s['N_f_cycles']:,} cycles",
            delta=s['risk'])

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Stress Distribution", "🌡️ Thermal Stack",
                                    "📉 Fatigue Life", "🔬 DOE Sweep"])

# ─── Tab 1: Stress ───────────────────────────────────────────────────────────
with tab1:
    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        stress = result.stress
        x = stress['x_arr_mm']
        tau = stress['tau_dist_MPa']

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, tau, color='#E24B4A', linewidth=2.5, label='Shear stress τ(x)')
        ax.fill_between(x, tau, alpha=0.15, color='#E24B4A')
        ax.axvline(x[-1], color='#378ADD', linestyle=':', linewidth=2, label='Die edge')
        ax.axhline(50, color='orange', linestyle='--', linewidth=1, label='ELK risk threshold (~50 MPa)')
        ax.axhline(stress['tau_max_MPa'], color='gray', linestyle='--', linewidth=1,
                   label=f'τ_max = {stress["tau_max_MPa"]:.1f} MPa')
        ax.set_xlabel('Distance from die center [mm]')
        ax.set_ylabel('Interfacial shear stress [MPa]')
        ax.set_title('CTE Mismatch Shear Stress (Suhir Model)', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_r:
        st.markdown("#### Stress Summary")
        st.markdown(f"""
| Parameter | Value |
|---|---|
| CTE mismatch | {stress['d_alpha_ppm']:.1f} ppm/°C |
| Characteristic length λ⁻¹ | {stress['characteristic_length_mm']:.2f} mm |
| τ_max (die edge) | **{stress['tau_max_MPa']:.2f} MPa** |
| σ_peel (die corner) | **{stress['sigma_peel_MPa']:.2f} MPa** |
| Von Mises σ_vm | **{stress['sigma_vm_MPa']:.2f} MPa** |
| Corner shear strain γ | {stress['gamma_max']*100:.4f}% |
        """)
        st.markdown("#### Risk Assessment")
        if stress['tau_max_MPa'] < 30:
            st.success("✅ LOW RISK — Stress well below ELK fracture threshold. Standard underfill adequate.")
        elif stress['tau_max_MPa'] < 50:
            st.warning("⚠️ MEDIUM RISK — Approaching ELK threshold. Monitor with crack stop structures.")
        elif stress['tau_max_MPa'] < 80:
            st.error("❌ HIGH RISK — Above ELK risk threshold. Low-CTE substrate or dual crack stop required.")
        else:
            st.error("🚨 CRITICAL — Severe CPI risk. Redesign substrate CTE and underfill stack immediately.")

        st.markdown("#### Model Reference")
        st.caption("Suhir, E. (1986). *Stresses in bi-metal thermostats.* J. Appl. Mech. 53, 657–660.")

# ─── Tab 2: Thermal ──────────────────────────────────────────────────────────
with tab2:
    thermal = result.thermal
    stack_mats  = [die, underfill, substrate]
    stack_thick = [die_thick, bump_height/1000, sub_t]
    stack_names = ['Silicon Die', 'Underfill', 'Substrate']

    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        fig, ax = plt.subplots(figsize=(5, 5))
        T_interfaces = thermal['T_interfaces_C'][:len(stack_thick)+1]
        z_vals = np.cumsum([0] + stack_thick)
        colors_stack = ['#4a90d9', '#e8c87a', '#8B6914']
        for i in range(len(stack_thick)):
            ax.fill_betweenx([z_vals[i], z_vals[i+1]],
                             T_interfaces[i], T_interfaces[min(i+1, len(T_interfaces)-1)],
                             alpha=0.5, color=colors_stack[i])
            mid_T = (T_interfaces[i] + T_interfaces[min(i+1, len(T_interfaces)-1)]) / 2
            ax.text(mid_T - 0.5, (z_vals[i]+z_vals[i+1])/2,
                    stack_names[i], ha='right', va='center', fontsize=9)
        ax.set_xlabel('Temperature [°C]')
        ax.set_ylabel('Depth from junction [mm]')
        ax.set_title('Steady-State Temperature Profile', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_r:
        st.markdown("#### Thermal Resistance Network")
        r_layers = thermal['R_layers_KW']
layer_names = ['Silicon Die', 'Underfill', 'Substrate', 'PCB']
rows = "\n".join([f"| {layer_names[i]} | {r:.4f} |"
                  for i, r in enumerate(r_layers)])
st.markdown(f"""
| Layer | R_th [K/W] |
|---|---|
{rows}
| **Total θ_ja** | **{thermal['R_total_KW']:.3f} K/W** |
""")
        st.metric("Junction Temperature", f"{thermal['T_junction_C']:.1f}°C")
        st.metric("Case Temperature", f"{thermal['T_case_C']:.1f}°C")
        st.metric("ΔT across stack", f"{thermal['delta_T_C']:.1f}°C")

# ─── Tab 3: Fatigue ──────────────────────────────────────────────────────────
with tab3:
    fatigue = result.fatigue
    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        die_sizes_plot = np.linspace(5, 25, 80)
        Nf_vals = []
        for ds in die_sizes_plot:
            g = PackageGeometry(die_size=ds, die_thickness=die_thick,
                                substrate_size=ds+7, substrate_thickness=sub_t,
                                bump_pitch=bump_pitch, bump_height=bump_height,
                                bump_diameter=int(bump_pitch*0.6),
                                underfill_thickness=bump_height, mold_thickness=0.0)
            Nf_vals.append(engelmaier_fatigue_life(g, die, substrate, profile)['N_f_cycles'])

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(die_sizes_plot, Nf_vals, color='#7F77DD', linewidth=2.5)
        ax.axhline(1000, color='orange', linestyle='--', linewidth=1.5, label='Consumer grade (1000)')
        ax.axhline(500,  color='red',    linestyle='--', linewidth=1.5, label='JEDEC minimum (500)')
        ax.axvline(die_size, color='#E24B4A', linestyle='-', linewidth=2, label=f'Current die ({die_size}mm)')
        ax.scatter([die_size], [fatigue['N_f_cycles']], color='#E24B4A', s=100, zorder=5)
        ax.set_xlabel('Die size [mm]')
        ax.set_ylabel('N_f [cycles]')
        ax.set_title('Solder Joint Fatigue Life vs Die Size', fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_r:
        st.markdown("#### Engelmaier Model Results")
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Shear strain range Δγ | {fatigue['delta_gamma']*100:.4f}% |
| Fatigue exponent c | {fatigue['c_exponent']:.4f} |
| Max DNP | {fatigue['DNP_mm']:.2f} mm |
| N_f (50% fail) | **{int(fatigue['N_f_cycles']):,} cycles** |
| Risk | **{fatigue['risk']}** |
        """)
        if fatigue['N_f_cycles'] > 2000:
            st.success("✅ Excellent — Automotive-grade reliability (>2000 cycles)")
        elif fatigue['N_f_cycles'] > 1000:
            st.success("✅ Good — Consumer-grade target met (>1000 cycles)")
        elif fatigue['N_f_cycles'] > 500:
            st.warning("⚠️ Marginal — Meets JEDEC minimum but not consumer grade")
        else:
            st.error("❌ FAIL — Below JEDEC TC-G minimum (500 cycles). Redesign required.")

# ─── Tab 4: DOE ──────────────────────────────────────────────────────────────
with tab4:
    st.info("Running 64-point DOE sweep (substrate CTE × underfill modulus × die size). This takes ~3 seconds.")
    if st.button("▶ Run DOE Sweep", type="primary"):
        with st.spinner("Computing..."):
            df_doe = doe_sweep(geometry, profile)
        st.success(f"Complete: {len(df_doe)} configurations evaluated")

        col_l, col_r = st.columns(2)
        with col_l:
            pivot = df_doe.groupby(['substrate_CTE', 'die_size_mm'])['tau_max_MPa'].mean().unstack()
            fig, ax = plt.subplots(figsize=(6, 4))
            import seaborn as sns
            sns.heatmap(pivot, ax=ax, cmap='RdYlGn_r', annot=True, fmt='.1f',
                        cbar_kws={'label': 'τ_max [MPa]'})
            ax.set_title('Max Shear Stress Heatmap\n(Substrate CTE × Die Size)', fontweight='bold')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_r:
            pivot2 = df_doe.groupby(['substrate_CTE', 'die_size_mm'])['N_f_cycles'].mean().unstack()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(pivot2, ax=ax, cmap='RdYlGn', annot=True, fmt='.0f',
                        cbar_kws={'label': 'N_f [cycles]'})
            ax.set_title('Fatigue Life Heatmap\n(Substrate CTE × Die Size)', fontweight='bold')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("#### Top 5 Configurations (lowest stress + highest life)")
        top5 = df_doe.nlargest(5, 'N_f_cycles')[
            ['substrate_CTE', 'underfill_E_GPa', 'die_size_mm',
             'tau_max_MPa', 'N_f_cycles', 'warpage_abs_um', 'risk']
        ].reset_index(drop=True)
        st.dataframe(top5.style.highlight_min(subset=['tau_max_MPa'], color='#d4edda')
                                .highlight_max(subset=['N_f_cycles'], color='#d4edda'),
                     use_container_width=True)
