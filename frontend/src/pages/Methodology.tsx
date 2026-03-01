import TeX from "@/components/TeX.tsx";
import AppLayout from "@/components/layout/AppLayout.tsx";

interface Section {
  title: string;
  items: { label: string; tex: string; note?: string }[];
}

const SECTIONS: Section[] = [
  {
    title: "Return Definitions",
    items: [
      {
        label: "Simple return",
        tex: "R_t = \\frac{P_t - P_{t-1}}{P_{t-1}}",
      },
      {
        label: "Log return",
        tex: "r_t = \\ln\\!\\left(\\frac{P_t}{P_{t-1}}\\right)",
        note: "Log returns are additive over time: r_{[t_0,\\,t_n]} = \\sum_{i=1}^{n} r_i",
      },
      {
        label: "Excess return",
        tex: "R^e_t = R_t - R^f_t",
      },
    ],
  },
  {
    title: "Signal Rules (Factor Strategy)",
    items: [
      {
        label: "Cross-sectional rank signal",
        tex: "w_{i,t} = \\begin{cases} +1/N_L & \\text{if } \\operatorname{rank}(f_{i,t}) \\le N_L \\\\ -1/N_S & \\text{if } \\operatorname{rank}(f_{i,t}) > N - N_S \\\\ 0 & \\text{otherwise} \\end{cases}",
        note: "N_L = \\lfloor N \\cdot p_{\\text{top}} \\rfloor, \\; N_S = \\lfloor N \\cdot p_{\\text{bottom}} \\rfloor",
      },
      {
        label: "Rebalancing",
        tex: "\\text{Positions update at } t \\in \\mathcal{T}_{\\text{rebal}} \\subset \\{t_1, t_2, \\ldots\\}",
      },
    ],
  },
  {
    title: "Performance Metrics",
    items: [
      {
        label: "Sharpe ratio",
        tex: "S = \\frac{\\mathbb{E}[R_p - R_f]}{\\sigma(R_p - R_f)} \\cdot \\sqrt{252}",
      },
      {
        label: "Sortino ratio",
        tex: "\\text{Sortino} = \\frac{\\mathbb{E}[R_p - R_f]}{\\sigma_d} \\cdot \\sqrt{252}, \\quad \\sigma_d = \\sqrt{\\frac{1}{N}\\sum_{r_t < 0} r_t^2}",
      },
      {
        label: "Maximum drawdown",
        tex: "\\text{MDD} = \\max_{t} \\left( \\frac{\\max_{s \\le t} V_s - V_t}{\\max_{s \\le t} V_s} \\right)",
      },
      {
        label: "Calmar ratio",
        tex: "\\text{Calmar} = \\frac{\\bar{R}_{\\text{ann}}}{|\\text{MDD}|}",
      },
      {
        label: "PnL with transaction costs",
        tex: "\\text{PnL}_t = \\mathbf{w}_{t-1}^\\top \\mathbf{r}_t - c \\cdot \\| \\mathbf{w}_t - \\mathbf{w}_{t-1} \\|_1",
        note: "c = cost per unit turnover (e.g. 10 bps)",
      },
    ],
  },
  {
    title: "Walk-Forward Protocol",
    items: [
      {
        label: "Expanding window split",
        tex: "\\mathcal{D}_{\\text{train}}^{(k)} = \\{t_0, \\ldots, t_0 + T_{\\text{init}} + k \\cdot T_{\\text{step}}\\}, \\quad \\mathcal{D}_{\\text{test}}^{(k)} = \\{t_{\\text{end}}^{(k)} + 1, \\ldots, t_{\\text{end}}^{(k)} + T_{\\text{test}}\\}",
      },
      {
        label: "Fold accuracy",
        tex: "\\text{Acc}^{(k)} = \\frac{1}{|\\mathcal{D}_{\\text{test}}^{(k)}|} \\sum_{t \\in \\mathcal{D}_{\\text{test}}^{(k)}} \\mathbb{1}[\\hat{y}_t = y_t]",
      },
      {
        label: "Overall accuracy",
        tex: "\\text{Acc} = \\frac{1}{K} \\sum_{k=1}^{K} \\text{Acc}^{(k)}",
        note: "K = number of walk-forward folds",
      },
    ],
  },
  {
    title: "Sortino Momentum Strategy",
    items: [
      {
        label: "Rolling Sortino slope",
        tex: "\\Delta S_x(t) = \\frac{S(t) - S(t-x)}{x}",
      },
      {
        label: "Strong momentum condition",
        tex: "\\Delta S_x(t) > \\Delta S_{30}(t - x)",
        note: "Recent slope exceeds the baseline slope computed over the prior 30-day window",
      },
      {
        label: "Hit rate",
        tex: "Z = \\frac{\\#\\{t : \\Delta S_k(t+k) > 0 \\mid \\text{strong momentum at } t\\}}{\\#\\{t : \\text{strong momentum at } t\\}} \\times 100",
      },
      {
        label: "Bootstrap p-value",
        tex: "p = \\frac{1}{B} \\sum_{b=1}^{B} \\mathbb{1}\\!\\left[|Z_b^* - \\bar{Z}^*| \\ge |Z_{\\text{obs}} - \\bar{Z}^*|\\right]",
      },
    ],
  },
];

export default function Methodology() {
  return (
    <AppLayout>
      <div className="max-w-3xl mx-auto py-6 px-4 space-y-10">
        <header>
          <h1 className="text-lg font-semibold text-zinc-200 tracking-tight">
            Methodology &amp; Equations
          </h1>
          <p className="text-xs text-zinc-500 mt-1">
            Mathematical definitions underlying the platform's strategies and metrics.
          </p>
        </header>

        {SECTIONS.map((section) => (
          <section key={section.title}>
            <h2 className="text-sm font-medium text-zinc-300 border-b border-zinc-800 pb-1 mb-4">
              {section.title}
            </h2>
            <div className="space-y-5">
              {section.items.map((item) => (
                <div key={item.label}>
                  <div className="text-[11px] uppercase tracking-wider text-zinc-500 mb-1">
                    {item.label}
                  </div>
                  <TeX math={item.tex} display />
                  {item.note && (
                    <div className="text-[11px] text-zinc-600 mt-1 pl-1">
                      <TeX math={item.note} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
    </AppLayout>
  );
}
