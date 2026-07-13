import { AnimatePresence, motion } from "framer-motion";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";

import DataCoverage from "./pages/DataCoverage.tsx";
import EconomicIndicators from "./pages/EconomicIndicators.tsx";
import Portfolio from "./pages/Portfolio.tsx";
import ExcludedStocks from "./pages/ExcludedStocks.tsx";
import FamaFrench from "./pages/FamaFrench.tsx";
import LinearAlgebra from "./pages/LinearAlgebra.tsx";
import MetalsAnalytics from "./pages/MetalsAnalytics.tsx";
import Methodology from "./pages/Methodology.tsx";
import MLAlphaReplay from "./pages/MLAlphaReplay.tsx";
import PairsTrading from "./pages/PairsTrading.tsx";
import PortfolioSimulator from "./pages/PortfolioSimulator.tsx";
import SectorBreakdown from "./pages/SectorBreakdown.tsx";
import SharpeRatioLimitations from "./pages/SharpeRatioLimitations.tsx";
import SortinoMomentum from "./pages/SortinoMomentum.tsx";

const pageVariants = {
  initial: { opacity: 0, y: 6 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.2, ease: "easeOut" } },
  exit: { opacity: 0, y: -6, transition: { duration: 0.12 } },
};

function AnimatedPage({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="h-full"
    >
      {children}
    </motion.div>
  );
}

export default function App() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<AnimatedPage><PortfolioSimulator /></AnimatedPage>} />
        <Route path="/ml-alpha" element={<AnimatedPage><MLAlphaReplay /></AnimatedPage>} />
        <Route path="/pairs" element={<AnimatedPage><PairsTrading /></AnimatedPage>} />
        <Route path="/momentum" element={<AnimatedPage><SortinoMomentum /></AnimatedPage>} />
        <Route path="/portfolio" element={<AnimatedPage><Portfolio /></AnimatedPage>} />
        <Route path="/etf-optimizer" element={<Navigate to="/portfolio" replace />} />
        <Route path="/manual-portfolio" element={<Navigate to="/portfolio" replace />} />
        <Route path="/metals" element={<AnimatedPage><MetalsAnalytics /></AnimatedPage>} />
        <Route path="/economic" element={<AnimatedPage><EconomicIndicators /></AnimatedPage>} />
        <Route path="/sectors" element={<AnimatedPage><SectorBreakdown /></AnimatedPage>} />
        <Route path="/excluded-stocks" element={<AnimatedPage><ExcludedStocks /></AnimatedPage>} />
        <Route path="/data-coverage" element={<AnimatedPage><DataCoverage /></AnimatedPage>} />
        <Route path="/methodology" element={<AnimatedPage><Methodology /></AnimatedPage>} />
        <Route path="/fama-french" element={<AnimatedPage><FamaFrench /></AnimatedPage>} />
        <Route path="/sharpe-limitations" element={<AnimatedPage><SharpeRatioLimitations /></AnimatedPage>} />
        <Route path="/linear-algebra" element={<AnimatedPage><LinearAlgebra /></AnimatedPage>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AnimatePresence>
  );
}
