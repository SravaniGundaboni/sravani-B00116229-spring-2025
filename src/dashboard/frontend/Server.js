
const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');

const app = express();
app.use(cors());

// Create MySQL connection
const connection = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: '',
    database: 'Sravani'
});
// Connect to MySQL
connection.connect((err) => {
    if (err) {
        console.error('Error connecting to MySQL:', err);
        return;
    }
    console.log('Connected to MySQL database');
});

// API endpoint to fetch flagged transactions
app.get('/api/transactions', (req, res) => {
    connection.query('SELECT * FROM transactions', (err, results) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        res.json(results);
    });
});
// API endpoint to fetch daily fraud counts
app.get('/api/stats/daily', (req, res) => {
    const query = `
        SELECT Time as time, COUNT(*) as fraud_count
        FROM transactions
        WHERE Prediction = 1
        GROUP BY time
        ORDER BY time
    `;
    connection.query(query, (err, results) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        res.json(results);
    });
});
// API endpoint to fetch fraud count
app.get('/api/stats/total', (req, res) => {
    connection.query('SELECT COUNT(*) as fraud_count FROM transactions WHERE Prediction = 1', (err, results) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        res.json(results[0]);
    });
});

// Start the server
const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
