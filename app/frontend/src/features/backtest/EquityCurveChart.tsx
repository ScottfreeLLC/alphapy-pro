import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid,
} from 'recharts';
import { EquityPoint } from '../../types/backtest';

interface EquityCurveChartProps {
  data: EquityPoint[];
  initialCapital: number;
}

export default function EquityCurveChart({ data, initialCapital }: EquityCurveChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        No equity data
      </div>
    );
  }

  const minEquity = Math.min(...data.map((d) => d.equity));
  const maxEquity = Math.max(...data.map((d) => d.equity));
  const yMin = Math.floor(minEquity * 0.98);
  const yMax = Math.ceil(maxEquity * 1.02);

  const formatCurrency = (value: number) =>
    `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

  return (
    <ResponsiveContainer width="100%" height={320}>
      <AreaChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
        <defs>
          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          dataKey="date"
          tick={{ fill: '#9ca3af', fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: '#4b5563' }}
          interval="preserveStartEnd"
          minTickGap={60}
        />
        <YAxis
          domain={[yMin, yMax]}
          tick={{ fill: '#9ca3af', fontSize: 11 }}
          tickFormatter={formatCurrency}
          tickLine={false}
          axisLine={{ stroke: '#4b5563' }}
          width={80}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1f2937',
            border: '1px solid #374151',
            borderRadius: '8px',
            color: '#f3f4f6',
          }}
          formatter={(value) => [formatCurrency(value as number), 'Equity']}
          labelStyle={{ color: '#9ca3af' }}
        />
        <ReferenceLine
          y={initialCapital}
          stroke="#6b7280"
          strokeDasharray="5 5"
          label={{
            value: 'Initial',
            position: 'right',
            fill: '#6b7280',
            fontSize: 11,
          }}
        />
        <Area
          type="monotone"
          dataKey="equity"
          stroke="#22c55e"
          strokeWidth={2}
          fill="url(#equityGradient)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
