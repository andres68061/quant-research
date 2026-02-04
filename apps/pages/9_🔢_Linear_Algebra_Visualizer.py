"""
Linear Algebra Visualizer for Quantitative Finance.

Interactive tool for learning matrix operations with applications to:
- Portfolio optimization
- Factor models
- Risk analysis
- Covariance matrices
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Page configuration
st.set_page_config(
    page_title="Linear Algebra Visualizer",
    page_icon="🔢",
    layout="wide",
)

st.title("🔢 Linear Algebra Visualizer")
st.markdown("""
Learn linear algebra through interactive visualizations with applications to quantitative finance.
""")

# ============================================================================
# SIDEBAR: MODE SELECTION
# ============================================================================

st.sidebar.header("📚 Learning Mode")

mode = st.sidebar.selectbox(
    "Choose a topic",
    [
        "2D Matrix Multiplication",
        "3D Geometric Transformations",
        "Portfolio Variance (Quadratic Form)",
        "Covariance Matrix",
        "Factor Models",
        "Matrix Properties"
    ]
)

# ============================================================================
# MODE 1: 2D MATRIX MULTIPLICATION
# ============================================================================

if mode == "2D Matrix Multiplication":
    st.header("📐 2D Matrix Multiplication")
    
    st.markdown("""
    **Learn how matrix multiplication works step by step.**
    
    For matrices A (m×n) and B (n×p), the result C = A × B is (m×p).
    Each element C[i,j] = sum of A[i,:] · B[:,j] (dot product of row i and column j).
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Matrix A")
        m = st.slider("Rows (m)", 1, 5, 2, key="m")
        n = st.slider("Columns (n)", 1, 5, 3, key="n")
        
        st.markdown("**Enter values for Matrix A:**")
        A = np.zeros((m, n))
        
        for i in range(m):
            cols = st.columns(n)
            for j in range(n):
                A[i, j] = cols[j].number_input(
                    f"A[{i},{j}]",
                    value=float(np.random.randint(1, 10)),
                    key=f"A_{i}_{j}",
                    label_visibility="collapsed"
                )
    
    with col2:
        st.subheader("Matrix B")
        p = st.slider("Columns (p)", 1, 5, 2, key="p")
        
        st.markdown(f"**Enter values for Matrix B:** (must have {n} rows)")
        B = np.zeros((n, p))
        
        for i in range(n):
            cols = st.columns(p)
            for j in range(p):
                B[i, j] = cols[j].number_input(
                    f"B[{i},{j}]",
                    value=float(np.random.randint(1, 10)),
                    key=f"B_{i}_{j}",
                    label_visibility="collapsed"
                )
    
    # Calculate result
    C = A @ B
    
    st.markdown("---")
    st.subheader("Result: C = A × B")
    
    # Display matrices side by side
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("**Matrix A**")
        st.dataframe(
            pd.DataFrame(A).style.format("{:.2f}"),
            use_container_width=True
        )
        st.caption(f"Shape: {A.shape}")
    
    with col2:
        st.markdown("**Matrix B**")
        st.dataframe(
            pd.DataFrame(B).style.format("{:.2f}"),
            use_container_width=True
        )
        st.caption(f"Shape: {B.shape}")
    
    with col3:
        st.markdown("**Result C**")
        st.dataframe(
            pd.DataFrame(C).style.format("{:.2f}").background_gradient(cmap='RdYlGn'),
            use_container_width=True
        )
        st.caption(f"Shape: {C.shape}")
    
    # Step-by-step calculation
    st.markdown("---")
    st.subheader("📝 Step-by-Step Calculation")
    
    result_i = st.slider("Select result row (i)", 0, m-1, 0)
    result_j = st.slider("Select result column (j)", 0, p-1, 0)
    
    st.markdown(f"### Calculating C[{result_i},{result_j}]")
    
    # Show the calculation
    row_A = A[result_i, :]
    col_B = B[:, result_j]
    
    calculation_steps = []
    running_sum = 0
    
    for k in range(n):
        product = row_A[k] * col_B[k]
        running_sum += product
        calculation_steps.append({
            'Step': k + 1,
            f'A[{result_i},{k}]': row_A[k],
            f'B[{k},{result_j}]': col_B[k],
            'Product': product,
            'Running Sum': running_sum
        })
    
    st.dataframe(
        pd.DataFrame(calculation_steps).style.format({
            f'A[{result_i},{k}]': '{:.2f}' for k in range(n)
        }).background_gradient(subset=['Running Sum'], cmap='Blues'),
        use_container_width=True
    )
    
    st.success(f"**Final Result: C[{result_i},{result_j}] = {C[result_i, result_j]:.2f}**")
    
    # Visual representation
    st.markdown("---")
    st.subheader("🎨 Visual Representation")
    
    # Create heatmaps
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Matrix A", "Matrix B", "Result C"),
        horizontal_spacing=0.15
    )
    
    # Matrix A heatmap
    fig.add_trace(
        go.Heatmap(
            z=A,
            colorscale='Blues',
            showscale=False,
            text=A,
            texttemplate='%{text:.1f}',
            textfont={"size": 14}
        ),
        row=1, col=1
    )
    
    # Matrix B heatmap
    fig.add_trace(
        go.Heatmap(
            z=B,
            colorscale='Greens',
            showscale=False,
            text=B,
            texttemplate='%{text:.1f}',
            textfont={"size": 14}
        ),
        row=1, col=2
    )
    
    # Result C heatmap
    fig.add_trace(
        go.Heatmap(
            z=C,
            colorscale='RdYlGn',
            showscale=True,
            text=C,
            texttemplate='%{text:.1f}',
            textfont={"size": 14}
        ),
        row=1, col=3
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MODE 2: 3D GEOMETRIC TRANSFORMATIONS
# ============================================================================

elif mode == "3D Geometric Transformations":
    st.header("🌐 3D Geometric Transformations")
    
    st.markdown("""
    **See how matrices transform vectors in 3D space.**
    
    A 3×3 matrix can represent rotations, scaling, shearing, and other transformations.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Transformation Type")
        
        transform_type = st.selectbox(
            "Choose transformation",
            ["Rotation (X-axis)", "Rotation (Y-axis)", "Rotation (Z-axis)", 
             "Scaling", "Shearing", "Custom Matrix"]
        )
        
        if "Rotation" in transform_type:
            angle = st.slider("Rotation Angle (degrees)", 0, 360, 45)
            theta = np.radians(angle)
            
            if "X-axis" in transform_type:
                T = np.array([
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]
                ])
                st.latex(r"""
                R_x(\theta) = \begin{bmatrix}
                1 & 0 & 0 \\
                0 & \cos\theta & -\sin\theta \\
                0 & \sin\theta & \cos\theta
                \end{bmatrix}
                """)
            
            elif "Y-axis" in transform_type:
                T = np.array([
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]
                ])
                st.latex(r"""
                R_y(\theta) = \begin{bmatrix}
                \cos\theta & 0 & \sin\theta \\
                0 & 1 & 0 \\
                -\sin\theta & 0 & \cos\theta
                \end{bmatrix}
                """)
            
            else:  # Z-axis
                T = np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                st.latex(r"""
                R_z(\theta) = \begin{bmatrix}
                \cos\theta & -\sin\theta & 0 \\
                \sin\theta & \cos\theta & 0 \\
                0 & 0 & 1
                \end{bmatrix}
                """)
        
        elif transform_type == "Scaling":
            sx = st.slider("Scale X", 0.1, 3.0, 1.0, 0.1)
            sy = st.slider("Scale Y", 0.1, 3.0, 1.0, 0.1)
            sz = st.slider("Scale Z", 0.1, 3.0, 1.0, 0.1)
            
            T = np.array([
                [sx, 0, 0],
                [0, sy, 0],
                [0, 0, sz]
            ])
            
            st.latex(r"""
            S = \begin{bmatrix}
            s_x & 0 & 0 \\
            0 & s_y & 0 \\
            0 & 0 & s_z
            \end{bmatrix}
            """)
        
        elif transform_type == "Shearing":
            shear = st.slider("Shear factor", -1.0, 1.0, 0.5, 0.1)
            
            T = np.array([
                [1, shear, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            
            st.latex(r"""
            H = \begin{bmatrix}
            1 & h & 0 \\
            0 & 1 & 0 \\
            0 & 0 & 1
            \end{bmatrix}
            """)
        
        else:  # Custom
            st.markdown("**Enter transformation matrix:**")
            T = np.zeros((3, 3))
            for i in range(3):
                cols = st.columns(3)
                for j in range(3):
                    T[i, j] = cols[j].number_input(
                        f"T[{i},{j}]",
                        value=1.0 if i == j else 0.0,
                        key=f"T_{i}_{j}",
                        label_visibility="collapsed"
                    )
        
        st.markdown("**Transformation Matrix:**")
        st.dataframe(
            pd.DataFrame(T).style.format("{:.3f}"),
            use_container_width=True
        )
    
    with col2:
        st.subheader("3D Visualization")
        
        # Create basis vectors
        origin = np.array([[0, 0, 0]])
        basis_vectors = np.array([
            [1, 0, 0],  # X-axis (red)
            [0, 1, 0],  # Y-axis (green)
            [0, 0, 1]   # Z-axis (blue)
        ])
        
        # Transform basis vectors
        transformed_vectors = (T @ basis_vectors.T).T
        
        # Create 3D plot
        fig = go.Figure()
        
        colors = ['red', 'green', 'blue']
        names = ['X-axis', 'Y-axis', 'Z-axis']
        
        # Original basis vectors
        for i, (vec, color, name) in enumerate(zip(basis_vectors, colors, names)):
            fig.add_trace(go.Scatter3d(
                x=[0, vec[0]],
                y=[0, vec[1]],
                z=[0, vec[2]],
                mode='lines+markers',
                line=dict(color=color, width=4, dash='dash'),
                marker=dict(size=[0, 8]),
                name=f'Original {name}',
                showlegend=True
            ))
        
        # Transformed basis vectors
        for i, (vec, color, name) in enumerate(zip(transformed_vectors, colors, names)):
            fig.add_trace(go.Scatter3d(
                x=[0, vec[0]],
                y=[0, vec[1]],
                z=[0, vec[2]],
                mode='lines+markers',
                line=dict(color=color, width=6),
                marker=dict(size=[0, 10]),
                name=f'Transformed {name}',
                showlegend=True
            ))
        
        # Add a cube to show transformation
        cube_vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
        ])
        
        transformed_cube = (T @ cube_vertices.T).T
        
        # Draw cube edges (original)
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top
            [0, 4], [1, 5], [2, 6], [3, 7]   # Sides
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=cube_vertices[edge, 0],
                y=cube_vertices[edge, 1],
                z=cube_vertices[edge, 2],
                mode='lines',
                line=dict(color='gray', width=2, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Draw transformed cube edges
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=transformed_cube[edge, 0],
                y=transformed_cube[edge, 1],
                z=transformed_cube[edge, 2],
                mode='lines',
                line=dict(color='purple', width=3),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-3, 3], title='X'),
                yaxis=dict(range=[-3, 3], title='Y'),
                zaxis=dict(range=[-3, 3], title='Z'),
                aspectmode='cube'
            ),
            height=600,
            showlegend=True,
            legend=dict(x=0.7, y=0.9)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **💡 Tip:** 
        - Dashed lines = original vectors/cube
        - Solid lines = transformed vectors/cube
        - Watch how the transformation changes the shape!
        """)

# ============================================================================
# MODE 3: PORTFOLIO VARIANCE (QUADRATIC FORM)
# ============================================================================

elif mode == "Portfolio Variance (Quadratic Form)":
    st.header("💼 Portfolio Variance: w^T Σ w")
    
    st.markdown("""
    **The most important formula in portfolio theory!**
    
    Portfolio variance is calculated as: **σ²_p = w^T Σ w**
    
    Where:
    - **w** = vector of portfolio weights (sums to 1)
    - **Σ** = covariance matrix of asset returns
    - **w^T** = transpose of w (row vector)
    
    This is a **quadratic form** - it measures how much risk (variance) your portfolio has.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Portfolio Setup")
        
        n_assets = st.slider("Number of Assets", 2, 5, 3)
        
        st.markdown("**Asset Weights (must sum to 1):**")
        weights = []
        remaining = 1.0
        
        for i in range(n_assets - 1):
            w = st.slider(
                f"Asset {i+1} weight",
                0.0,
                remaining,
                remaining / (n_assets - i),
                0.01,
                key=f"w_{i}"
            )
            weights.append(w)
            remaining -= w
        
        weights.append(max(0, remaining))
        st.caption(f"Asset {n_assets} weight: {weights[-1]:.2f} (auto-calculated)")
        
        w = np.array(weights)
        
        st.markdown("**Covariance Matrix:**")
        
        # Generate a realistic covariance matrix
        if st.button("Generate Random Covariance Matrix"):
            st.session_state['cov_matrix'] = None
        
        if 'cov_matrix' not in st.session_state or st.session_state['cov_matrix'] is None:
            # Generate random correlation matrix
            A = np.random.randn(n_assets, n_assets)
            corr = A @ A.T
            corr = corr / np.sqrt(np.diag(corr)[:, None] @ np.diag(corr)[None, :])
            
            # Add volatilities
            vols = np.random.uniform(0.15, 0.35, n_assets)  # 15-35% annual vol
            Sigma = np.diag(vols) @ corr @ np.diag(vols)
            
            st.session_state['cov_matrix'] = Sigma
        else:
            Sigma = st.session_state['cov_matrix']
        
        st.dataframe(
            pd.DataFrame(
                Sigma,
                columns=[f"Asset {i+1}" for i in range(n_assets)],
                index=[f"Asset {i+1}" for i in range(n_assets)]
            ).style.format("{:.4f}").background_gradient(cmap='RdYlGn_r'),
            use_container_width=True
        )
        
        # Calculate portfolio variance
        portfolio_variance = w.T @ Sigma @ w
        portfolio_vol = np.sqrt(portfolio_variance)
        
        st.success(f"**Portfolio Variance:** {portfolio_variance:.6f}")
        st.success(f"**Portfolio Volatility:** {portfolio_vol:.4f} ({portfolio_vol*100:.2f}%)")
    
    with col2:
        st.subheader("Breakdown of Calculation")
        
        st.markdown("### Step 1: w^T (transpose)")
        st.code(f"w^T = {w}")
        
        st.markdown("### Step 2: Σ × w (matrix-vector multiplication)")
        Sigma_w = Sigma @ w
        st.code(f"Σ × w = {Sigma_w}")
        
        st.markdown("### Step 3: w^T × (Σ × w) (dot product)")
        st.code(f"w^T × (Σ × w) = {portfolio_variance:.6f}")
        
        st.markdown("---")
        st.subheader("Risk Contribution by Asset Pair")
        
        # Calculate contribution of each pair
        contributions = np.outer(w, w) * Sigma
        
        fig = go.Figure(data=go.Heatmap(
            z=contributions,
            x=[f"Asset {i+1}" for i in range(n_assets)],
            y=[f"Asset {i+1}" for i in range(n_assets)],
            colorscale='RdYlGn_r',
            text=contributions,
            texttemplate='%{text:.4f}',
            textfont={"size": 10},
            colorbar=dict(title="Contribution")
        ))
        
        fig.update_layout(
            title="Risk Contribution Matrix (w_i × w_j × σ_ij)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Interpretation:**")
        st.info("""
        - **Diagonal elements:** Variance contribution from each asset alone
        - **Off-diagonal elements:** Covariance contribution from asset pairs
        - **Sum of all elements = Portfolio Variance**
        
        💡 Diversification works because off-diagonal elements can be negative (negative correlation)!
        """)
        
        # Individual contributions
        st.markdown("### Individual Asset Risk Contributions")
        
        marginal_contrib = Sigma @ w
        contrib_pct = (w * marginal_contrib) / portfolio_variance * 100
        
        contrib_df = pd.DataFrame({
            'Asset': [f"Asset {i+1}" for i in range(n_assets)],
            'Weight (%)': w * 100,
            'Marginal Contribution': marginal_contrib,
            'Total Contribution (%)': contrib_pct
        })
        
        st.dataframe(
            contrib_df.style.format({
                'Weight (%)': '{:.2f}%',
                'Marginal Contribution': '{:.4f}',
                'Total Contribution (%)': '{:.2f}%'
            }).background_gradient(subset=['Total Contribution (%)'], cmap='RdYlGn_r'),
            use_container_width=True
        )

# ============================================================================
# MODE 4: COVARIANCE MATRIX
# ============================================================================

elif mode == "Covariance Matrix":
    st.header("📊 Covariance Matrix Visualization")
    
    st.markdown("""
    **Understanding the covariance matrix is crucial for portfolio optimization.**
    
    The covariance matrix Σ captures:
    - **Diagonal:** Variance of each asset (σ²_i)
    - **Off-diagonal:** Covariance between assets (σ_ij)
    
    Formula: **Σ = (1/n) × X^T × X** where X is the centered returns matrix.
    """)
    
    # Load actual stock data
    try:
        prices = pd.read_parquet(ROOT / "data" / "factors" / "prices.parquet")
        
        # Get a few stocks
        available_stocks = [col for col in prices.columns if not col.startswith('^')][:50]
        
        selected_stocks = st.multiselect(
            "Select stocks to analyze",
            available_stocks,
            default=available_stocks[:5]
        )
        
        if len(selected_stocks) >= 2:
            # Calculate returns
            stock_prices = prices[selected_stocks].dropna()
            returns = stock_prices.pct_change().dropna()
            
            # Calculate covariance matrix (annualized)
            cov_matrix = returns.cov() * 252
            corr_matrix = returns.corr()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Covariance Matrix (Annualized)")
                
                fig_cov = go.Figure(data=go.Heatmap(
                    z=cov_matrix.values,
                    x=selected_stocks,
                    y=selected_stocks,
                    colorscale='RdYlGn_r',
                    text=cov_matrix.values,
                    texttemplate='%{text:.4f}',
                    textfont={"size": 8},
                    colorbar=dict(title="Covariance")
                ))
                
                fig_cov.update_layout(height=500)
                st.plotly_chart(fig_cov, use_container_width=True)
                
                # Show volatilities
                vols = np.sqrt(np.diag(cov_matrix))
                vol_df = pd.DataFrame({
                    'Stock': selected_stocks,
                    'Volatility': vols,
                    'Volatility (%)': vols * 100
                })
                
                st.dataframe(
                    vol_df.style.format({
                        'Volatility': '{:.4f}',
                        'Volatility (%)': '{:.2f}%'
                    }).background_gradient(subset=['Volatility (%)'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Correlation Matrix")
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=selected_stocks,
                    y=selected_stocks,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 8},
                    colorbar=dict(title="Correlation")
                ))
                
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                st.info("""
                **Correlation vs Covariance:**
                - **Correlation:** Standardized (-1 to +1)
                - **Covariance:** Actual units (σ_i × σ_j × ρ_ij)
                
                Relationship: **σ_ij = ρ_ij × σ_i × σ_j**
                """)
            
            # Eigenvalue decomposition
            st.markdown("---")
            st.subheader("📐 Eigenvalue Decomposition (Principal Components)")
            
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix.values)
            
            # Sort by eigenvalue
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Variance explained
            var_explained = eigenvalues / eigenvalues.sum() * 100
            cumvar_explained = np.cumsum(var_explained)
            
            fig_eigen = go.Figure()
            
            fig_eigen.add_trace(go.Bar(
                x=[f"PC{i+1}" for i in range(len(eigenvalues))],
                y=var_explained,
                name='Variance Explained',
                marker_color='lightblue'
            ))
            
            fig_eigen.add_trace(go.Scatter(
                x=[f"PC{i+1}" for i in range(len(eigenvalues))],
                y=cumvar_explained,
                name='Cumulative',
                mode='lines+markers',
                marker=dict(size=10, color='red'),
                yaxis='y2'
            ))
            
            fig_eigen.update_layout(
                title="Principal Component Analysis",
                xaxis_title="Principal Component",
                yaxis_title="Variance Explained (%)",
                yaxis2=dict(
                    title="Cumulative (%)",
                    overlaying='y',
                    side='right'
                ),
                height=400
            )
            
            st.plotly_chart(fig_eigen, use_container_width=True)
            
            st.markdown(f"""
            **Key Insight:** The first {np.argmax(cumvar_explained > 80) + 1} principal components 
            explain {cumvar_explained[np.argmax(cumvar_explained > 80)]:.1f}% of the variance!
            """)
        
        else:
            st.warning("Please select at least 2 stocks")
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure you have price data in data/factors/prices.parquet")

# ============================================================================
# MODE 5: FACTOR MODELS
# ============================================================================

elif mode == "Factor Models":
    st.header("📈 Factor Models: R = α + β × F")
    
    st.markdown("""
    **Factor models explain returns using common factors.**
    
    Single-factor model: **R_i = α_i + β_i × F + ε_i**
    
    Multi-factor model: **R_i = α_i + Σ(β_ij × F_j) + ε_i**
    
    In matrix form: **R = α + B × F + ε**
    
    Where:
    - **R** = asset returns (n × 1)
    - **α** = alpha (intercept)
    - **B** = factor loadings (n × k matrix)
    - **F** = factor returns (k × 1)
    - **ε** = idiosyncratic returns
    """)
    
    st.subheader("Example: 3-Factor Model")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Factor Loadings (β)**")
        
        n_assets = 5
        factors = ['Market', 'Size', 'Value']
        
        # Generate random factor loadings
        B = np.random.randn(n_assets, len(factors))
        B[:, 0] = np.abs(B[:, 0])  # Market beta typically positive
        
        B_df = pd.DataFrame(
            B,
            columns=factors,
            index=[f"Stock {i+1}" for i in range(n_assets)]
        )
        
        st.dataframe(
            B_df.style.format("{:.3f}").background_gradient(cmap='RdYlGn'),
            use_container_width=True
        )
        
        st.markdown("**Factor Returns (F)**")
        
        F = np.array([
            st.slider("Market return (%)", -10.0, 10.0, 5.0, 0.1) / 100,
            st.slider("Size return (%)", -5.0, 5.0, 1.0, 0.1) / 100,
            st.slider("Value return (%)", -5.0, 5.0, 2.0, 0.1) / 100
        ])
        
        st.code(f"F = {F}")
    
    with col2:
        st.markdown("**Asset Returns Decomposition**")
        
        # Calculate returns
        alpha = np.random.randn(n_assets) * 0.01  # Small alphas
        factor_returns = B @ F
        epsilon = np.random.randn(n_assets) * 0.02  # Idiosyncratic
        total_returns = alpha + factor_returns + epsilon
        
        # Create decomposition chart
        decomp_df = pd.DataFrame({
            'Stock': [f"Stock {i+1}" for i in range(n_assets)],
            'Alpha': alpha,
            'Market': B[:, 0] * F[0],
            'Size': B[:, 1] * F[1],
            'Value': B[:, 2] * F[2],
            'Idiosyncratic': epsilon,
            'Total Return': total_returns
        })
        
        fig = go.Figure()
        
        components = ['Alpha', 'Market', 'Size', 'Value', 'Idiosyncratic']
        colors = ['gray', 'blue', 'green', 'orange', 'red']
        
        for component, color in zip(components, colors):
            fig.add_trace(go.Bar(
                name=component,
                x=decomp_df['Stock'],
                y=decomp_df[component] * 100,
                marker_color=color
            ))
        
        fig.add_trace(go.Scatter(
            name='Total Return',
            x=decomp_df['Stock'],
            y=decomp_df['Total Return'] * 100,
            mode='markers',
            marker=dict(size=15, color='black', symbol='diamond'),
            yaxis='y'
        ))
        
        fig.update_layout(
            barmode='relative',
            title="Return Decomposition by Factor",
            xaxis_title="Stock",
            yaxis_title="Return (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            decomp_df.style.format({
                col: '{:.2f}%' if col != 'Stock' else '{}'
                for col in decomp_df.columns
            }).background_gradient(subset=['Total Return'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        st.info("""
        **Interpretation:**
        - **Market factor:** Systematic risk (cannot diversify)
        - **Size/Value factors:** Style exposures
        - **Alpha:** Skill-based excess return
        - **Idiosyncratic:** Stock-specific risk (diversifiable)
        """)

# ============================================================================
# MODE 6: MATRIX PROPERTIES
# ============================================================================

elif mode == "Matrix Properties":
    st.header("🔍 Matrix Properties")
    
    st.markdown("""
    **Learn about important matrix properties used in quant finance.**
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Matrix")
        
        n = st.slider("Matrix size (n×n)", 2, 5, 3)
        
        st.markdown("**Enter matrix values:**")
        A = np.zeros((n, n))
        
        for i in range(n):
            cols = st.columns(n)
            for j in range(n):
                A[i, j] = cols[j].number_input(
                    f"A[{i},{j}]",
                    value=float(np.random.randint(-5, 6)),
                    key=f"prop_A_{i}_{j}",
                    label_visibility="collapsed"
                )
        
        st.dataframe(
            pd.DataFrame(A).style.format("{:.2f}"),
            use_container_width=True
        )
    
    with col2:
        st.subheader("Matrix Properties")
        
        # Transpose
        st.markdown("### 1. Transpose (A^T)")
        st.dataframe(
            pd.DataFrame(A.T).style.format("{:.2f}"),
            use_container_width=True
        )
        
        # Determinant
        st.markdown("### 2. Determinant")
        det_A = np.linalg.det(A)
        st.metric("det(A)", f"{det_A:.4f}")
        
        if abs(det_A) < 1e-10:
            st.warning("⚠️ Matrix is singular (not invertible)")
        else:
            st.success("✅ Matrix is invertible")
        
        # Trace
        st.markdown("### 3. Trace (sum of diagonal)")
        trace_A = np.trace(A)
        st.metric("tr(A)", f"{trace_A:.4f}")
        
        # Rank
        st.markdown("### 4. Rank")
        rank_A = np.linalg.matrix_rank(A)
        st.metric("rank(A)", rank_A)
        
        if rank_A == n:
            st.success("✅ Full rank")
        else:
            st.warning(f"⚠️ Rank deficient ({rank_A} < {n})")
        
        # Inverse (if exists)
        if abs(det_A) > 1e-10:
            st.markdown("### 5. Inverse (A^-1)")
            A_inv = np.linalg.inv(A)
            st.dataframe(
                pd.DataFrame(A_inv).style.format("{:.4f}"),
                use_container_width=True
            )
            
            # Verify A × A^-1 = I
            identity = A @ A_inv
            st.markdown("**Verification: A × A^-1 = I**")
            st.dataframe(
                pd.DataFrame(identity).style.format("{:.4f}").background_gradient(cmap='Greens'),
                use_container_width=True
            )
        
        # Eigenvalues
        st.markdown("### 6. Eigenvalues & Eigenvectors")
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        st.markdown("**Eigenvalues:**")
        st.code(eigenvalues)
        
        st.markdown("**Eigenvectors:**")
        st.dataframe(
            pd.DataFrame(eigenvectors).style.format("{:.4f}"),
            use_container_width=True
        )
        
        # Symmetric check
        st.markdown("### 7. Symmetry Check")
        is_symmetric = np.allclose(A, A.T)
        
        if is_symmetric:
            st.success("✅ Matrix is symmetric (A = A^T)")
            st.info("Symmetric matrices have real eigenvalues and orthogonal eigenvectors!")
        else:
            st.info("Matrix is not symmetric")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
💡 **Learning Tips:**
- Start with 2D matrix multiplication to understand the basics
- Move to 3D transformations to see geometric intuition
- Apply to portfolio variance to see real quant finance usage
- Experiment with different values to build intuition!

📚 **Resources:**
- [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Khan Academy - Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
- [MIT OpenCourseWare - Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
""")
