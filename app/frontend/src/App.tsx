import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import AgentDashboard from './features/agent/AgentDashboard';
import ScreenerPage from './features/screener/ScreenerPage';
import TradeProposals from './features/trades/TradeProposals';
import PortfolioView from './features/portfolio/PortfolioView';
import SkillsManager from './features/strategies/SkillsManager';
import BacktestPage from './features/backtest/BacktestPage';
import PublishPage from './features/publish/PublishPage';
import ChatPage from './features/chat/ChatPage';

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<AgentDashboard agentType="swing" />} />
        <Route path="/day" element={<AgentDashboard agentType="day" />} />
        <Route path="/screener" element={<ScreenerPage />} />
        <Route path="/trades" element={<TradeProposals />} />
        <Route path="/portfolio" element={<PortfolioView />} />
        <Route path="/strategies" element={<SkillsManager />} />
        <Route path="/backtest" element={<BacktestPage />} />
        <Route path="/publish" element={<PublishPage />} />
        <Route path="/chat" element={<ChatPage />} />
      </Route>
    </Routes>
  );
}

export default App;
