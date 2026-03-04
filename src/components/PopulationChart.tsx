"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

interface PopulationChartProps {
  currentPopulation: number;
  trend: "increasing" | "decreasing" | "stable" | "unknown";
  speciesName: string;
  className?: string;
}

/**
 * Generate simulated historical population data points.
 * Uses the current population + trend to extrapolate backwards realistically.
 */
function generateHistoricalData(
  current: number,
  trend: "increasing" | "decreasing" | "stable" | "unknown",
): { year: string; population: number }[] {
  const points: { year: string; population: number }[] = [];
  const currentYear = new Date().getFullYear();
  const years = 12;

  for (let i = years; i >= 0; i--) {
    const year = currentYear - i;
    let factor: number;

    switch (trend) {
      case "increasing":
        // Population was lower in the past, growing ~3-5% per year
        factor = Math.pow(0.96, i) + (Math.random() - 0.5) * 0.04;
        break;
      case "decreasing":
        // Population was higher in the past, declining ~2-4% per year
        factor = Math.pow(1.035, i) + (Math.random() - 0.5) * 0.04;
        break;
      case "stable":
        // Small random fluctuations around current value
        factor = 1 + (Math.random() - 0.5) * 0.08;
        break;
      default:
        factor = 1 + (Math.random() - 0.5) * 0.1;
    }

    points.push({
      year: year.toString(),
      population: Math.round(current * factor),
    });
  }

  return points;
}

export default function PopulationChart({
  currentPopulation,
  trend,
  speciesName,
  className = "",
}: PopulationChartProps) {
  const data = generateHistoricalData(currentPopulation, trend);

  const trendColor =
    trend === "increasing"
      ? "#10b981"
      : trend === "decreasing"
      ? "#ef4444"
      : "#2dd4bf";

  const minPop = Math.min(...data.map((d) => d.population));
  const maxPop = Math.max(...data.map((d) => d.population));
  const padding = Math.round((maxPop - minPop) * 0.1);

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
          <defs>
            <linearGradient id={`popGradient-${speciesName}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={trendColor} stopOpacity={0.15} />
              <stop offset="100%" stopColor={trendColor} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.04)"
            vertical={false}
          />
          <XAxis
            dataKey="year"
            tick={{ fill: "#71717a", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            interval={2}
          />
          <YAxis
            tick={{ fill: "#71717a", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            domain={[minPop - padding, maxPop + padding]}
            tickFormatter={(v: number) =>
              v >= 1000 ? `${(v / 1000).toFixed(1)}K` : v.toString()
            }
          />
          <Tooltip
            contentStyle={{
              background: "#18181b",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: 8,
              color: "#fff",
              fontSize: 12,
            }}
            formatter={(value) => [typeof value === 'number' ? new Intl.NumberFormat("en-US").format(value) : value, "Population"]}
            labelFormatter={(label) => `Year ${label}`}
          />
          <Line
            type="monotone"
            dataKey="population"
            stroke={trendColor}
            strokeWidth={2}
            dot={{ r: 2, fill: trendColor, strokeWidth: 0 }}
            activeDot={{ r: 4, fill: trendColor, stroke: "#09090b", strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
