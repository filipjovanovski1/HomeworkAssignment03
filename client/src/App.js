import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';

// REGISTER BASE PLUGINS
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

function App() {
    const [codes, setCodes] = useState([]);
    const [selectedCode, setSelectedCode] = useState('');
    const [transactions, setTransactions] = useState([]);

    // Fields we want to include in the chart (all numeric except "id")
    // Adjust as needed:
    const numericFields = [
        'close',
        'momentum',
        'sma20',
        'sma50',
        'ema20',
        'ema50',
        'bbmid',
        'rsi',
        'obv',
        'cci'
    ];

    // Fetch codes once on mount
    useEffect(() => {
        fetch('http://localhost:8080/api/get/codes')
            .then((res) => res.json())
            .then((data) => {
                setCodes(data);
                if (data.length > 0) {
                    setSelectedCode(data[0]); // default to first code
                }
            })
            .catch((err) => console.error('Error fetching codes:', err));
    }, []);

    // Fetch transactions for selected code
    const handleShowTransactions = () => {
        if (!selectedCode) return;
        fetch(`http://localhost:8080/api/get/transaction/${selectedCode}`)
            .then((res) => res.json())
            .then((data) => {
                setTransactions(data);
                console.log(data);
            })
            .catch((err) => console.error('Error fetching transactions:', err));
    };

    // Dynamically build a dataset for each numeric field
    const datasets = numericFields.map((field, idx) => {
        const colorHue = 360 * (idx / numericFields.length); // generate a unique hue
        const borderColor = `hsl(${colorHue}, 70%, 50%)`;
        const backgroundColor = `hsl(${colorHue}, 70%, 80%)`;
        return {
            label: field,
            data: transactions.map((tx) => Number(tx[field]) || 0),
            borderColor,
            backgroundColor,
            tension: 0.2,
            fill: false,
        };
    });

    // Prepare chart data
    const chartData = {
        labels: transactions.map((tx) => {
            const d = new Date(tx.date);
            return d.toLocaleDateString();
        }),
        datasets,
    };

    // CHART OPTIONS with decimation + reduced point radius + disabled animations
    const chartOptions = {
        responsive: true,
        animation: false,  // Turn off animations to improve performance
        plugins: {
            legend: { position: 'top' },
            title: {
                display: true,
                text: `Transactions for ${selectedCode || ''}`,
            },
            // Decimation plugin settings (built into Chart.js v3+)
            decimation: {
                enabled: true,
                algorithm: 'lttb',  // 'lttb' or 'min-max'
                samples: 200,       // how many points to keep
                threshold: 1000,    // apply decimation above this # of points
            },
        },
        elements: {
            point: {
                radius: 1,    // very small point radius
                hitRadius: 5, // optional, makes it easier to hover
            },
        },
    };

    return (
        <div style={{ maxWidth: '1100px', margin: '40px auto', fontFamily: 'Arial, sans-serif' }}>
            <h1 style={{ textAlign: 'center' }}>Company Transactions Viewer</h1>

            {/* Dropdown & button */}
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '20px' }}>
                <select
                    style={{ fontSize: '16px', padding: '6px' }}
                    value={selectedCode}
                    onChange={(e) => setSelectedCode(e.target.value)}
                >
                    {codes.map((code) => (
                        <option key={code} value={code}>
                            {code}
                        </option>
                    ))}
                </select>
                <button
                    style={{
                        marginLeft: '10px',
                        padding: '6px 12px',
                        fontSize: '16px',
                        backgroundColor: '#4CAF50',
                        color: '#fff',
                        border: 'none',
                        cursor: 'pointer'
                    }}
                    onClick={handleShowTransactions}
                >
                    Show Transactions
                </button>
            </div>

            {/* Chart */}
            {transactions.length > 0 && (
                <div style={{ margin: '20px 0' }}>
                    <Line data={chartData} options={chartOptions} />
                </div>
            )}

            {/* Transactions table */}
            {transactions.length > 0 && (
                <table
                    style={{
                        width: '100%',
                        borderCollapse: 'collapse',
                        marginTop: '20px',
                        fontSize: '14px'
                    }}
                >
                    <thead style={{ backgroundColor: '#f2f2f2' }}>
                    <tr>
                        <th style={thStyle}>Date</th>
                        <th style={thStyle}>Close</th>
                        <th style={thStyle}>Max</th>
                        <th style={thStyle}>Min</th>
                        <th style={thStyle}>AvgPrice</th>
                        <th style={thStyle}>%Change</th>
                        <th style={thStyle}>Volume</th>
                        <th style={thStyle}>Turnover In Best</th>
                        <th style={thStyle}>Total Turnover</th>
                        <th style={thStyle}>Momentum</th>
                        <th style={thStyle}>SMA20</th>
                        <th style={thStyle}>SMA50</th>
                        <th style={thStyle}>EMA20</th>
                        <th style={thStyle}>EMA50</th>
                        <th style={thStyle}>BBMid</th>
                        <th style={thStyle}>RSI</th>
                        <th style={thStyle}>OBV</th>
                        <th style={thStyle}>CCI</th>
                        <th style={thStyle}>Signal</th>
                        <th style={thStyle}>CCISignal</th>
                    </tr>
                    </thead>
                    <tbody>
                    {transactions.map((tx, idx) => {
                        const dateObj = new Date(tx.date);
                        return (
                            <tr key={idx} style={{ textAlign: 'center' }}>
                                <td style={tdStyle}>{dateObj.toLocaleDateString()}</td>
                                <td style={tdStyle}>{tx.close}</td>
                                <td style={tdStyle}>{tx.max}</td>
                                <td style={tdStyle}>{tx.min}</td>
                                <td style={tdStyle}>{tx.avgPrice}</td>
                                <td style={tdStyle}>{tx.perChange}</td>
                                <td style={tdStyle}>{tx.volume}</td>
                                <td style={tdStyle}>{tx.turnoverInBest}</td>
                                <td style={tdStyle}>{tx.totalTurnover}</td>
                                <td style={tdStyle}>{tx.momentum}</td>
                                <td style={tdStyle}>{tx.sma20}</td>
                                <td style={tdStyle}>{tx.sma50}</td>
                                <td style={tdStyle}>{tx.ema20}</td>
                                <td style={tdStyle}>{tx.ema50}</td>
                                <td style={tdStyle}>{tx.bbmid}</td>
                                <td style={tdStyle}>{tx.rsi}</td>
                                <td style={tdStyle}>{tx.obv}</td>
                                <td style={tdStyle}>{tx.cci}</td>
                                <td style={tdStyle}>{tx.signal}</td>
                                <td style={tdStyle}>{tx.ccisignal}</td>
                            </tr>
                        );
                    })}
                    </tbody>
                </table>
            )}
        </div>
    );
}

// Simple styling for table cells
const thStyle = {
    padding: '8px',
    border: '1px solid #ddd'
};
const tdStyle = {
    padding: '6px',
    border: '1px solid #ddd'
};

export default App;
