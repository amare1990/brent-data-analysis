import React from "react";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import OilData from "./components/OilData";
import ArimaResults from "./components/ArimaResults";
import Describe from "./components/Describe";
import ModelComparison from "./components/ModelComparison";

function App() {
    return (
        <Router>
            <div>
                <nav>
                    <ul>
                        <li><Link to="/">Brent Oil Data</Link></li>
                        <li><Link to="/describe">Summary Statistics </Link></li>
                        <li><Link to="/arima">ARIMA Summary</Link></li>
                        <li><Link to="/compare_models">Model Comparison</Link></li>
                    </ul>
                </nav>

                <Routes>
                    <Route path="/" element={<OilData />} />
                    <Route path="/describe" element={< Describe />}/>
                    <Route path="/arima" element={<ArimaResults />} />
                    <Route path="/compare_models" element={<ModelComparison />} />
                </Routes>
            </div>
        </Router>
    );
}

export default App;
