import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SetupMod from './pages/SetupMod';
import BrandPage from './pages/[brand]';

const App = () => {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<SetupMod />} />
                <Route path="/:brand" element={<BrandPage />} />
            </Routes>
        </Router>
    );
};

export default App;