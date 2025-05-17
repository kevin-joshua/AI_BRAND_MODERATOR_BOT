import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';
import config from '../config';
import './BrandPage.css';

const BrandPage = () => {
    const { brand } = useParams();
    const [response, setResponse] = useState(null);
    const [query, setQuery] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleInputChange = (e) => {
        setQuery(e.target.value);
        setError(null);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!query.trim()) {
            setError('Please enter a query');
            return;
        }

        setIsLoading(true);
        setError(null);
        
        try {
            const res = await axios.get(`${config.API_BASE_URL}${config.ENDPOINTS.QUERY}?brand=${brand}&query=${encodeURIComponent(query)}`, {
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            setResponse(res.data.answer);
        } catch (error) {
            setError(error.response?.data?.message || 'Error fetching data. Please try again.');
            setResponse(null);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="brand-page">
            <div className="brand-header">
                <h1>{brand}</h1>
                <p className="brand-description">Ask questions about {brand}, anything you want</p>
            </div>

            <div className="query-section">
                <form onSubmit={handleSubmit} className="query-form">
                    <div className="input-group">
                        <input
                            type="text"
                            value={query}
                            onChange={handleInputChange}
                            placeholder="Enter your query"
                            className="query-input"
                            disabled={isLoading}
                        />
                        <button 
                            type="submit" 
                            className="submit-button"
                            disabled={isLoading}
                        >
                            {isLoading ? 'Processing...' : 'Ask'}
                        </button>
                    </div>
                    {error && <div className="error-message">{error}</div>}
                </form>

                {response && (
                    <div className="response-container">
                        <h3>Response:</h3>
                        <div className="response-content">
                            {typeof response === 'string' ? response : JSON.stringify(response, null, 2)}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default BrandPage;