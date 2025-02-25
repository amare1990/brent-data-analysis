import React from "react";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import OilData from "./components/OilData";
import ArimaResults from "./components/ArimaResults";

function App() {
    return (
        <Router>
            <div>
                <nav>
                    <ul>
                        <li><Link to="/">Brent Oil Data</Link></li>
                        <li><Link to="/arima">ARIMA Summary</Link></li>
                    </ul>
                </nav>

                <Routes>
                    <Route path="/" element={<OilData />} />
                    <Route path="/arima" element={<ArimaResults />} />
                </Routes>
            </div>
        </Router>
    );
}

export default App;
