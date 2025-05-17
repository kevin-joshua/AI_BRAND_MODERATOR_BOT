import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import config from '../config';
import './FileUpload.css';

const FileUpload = ({ onSuccess, onError }) => {
    const [url, setUrl] = useState('');
    const [pdfFiles, setPdfFiles] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(false);
    const [brand, setBrand] = useState('');

    const navigate = useNavigate();
    const handleUploadSuccess = (brand) => {
        navigate(`/${brand}`);
    };

    useEffect(() => {
        if (brand) {
            handleUploadSuccess(brand);
        }
    }, [success]);

    const extractBrandFromUrl = (urlString) => {
        try {
            const url = new URL(urlString);
            // Get the hostname and remove www. if present
            let hostname = url.hostname.replace('www.', '');
            // Split by dots and get the first part (usually the brand name)
            return hostname.split('.')[0];
        } catch (e) {
            return null;
        }
    };

    const handleFileChange = (event) => {
        const files = Array.from(event.target.files || []);
        const pdfFiles = files.filter(file => file.type === 'application/pdf');
        
        if (pdfFiles.length !== files.length) {
            setError('Only PDF files are allowed');
            return;
        }
        
        setPdfFiles(pdfFiles);
        setError(null);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        
        if (!url && pdfFiles.length === 0) {
            setError('Please provide either a URL or upload PDF files');
            return;
        }

        setIsLoading(true);
        setError(null);
      

        try {
            const requests = [];

            // Handle URL submission if URL is present
            if (url) {
                const brandName = extractBrandFromUrl(url);
                
                if (!brandName) {
                    throw new Error('Invalid URL format');
                }
                setBrand(brandName);
                requests.push(
                    axios.post(`${config.API_BASE_URL}${config.ENDPOINTS.SCRAPE}?url=${encodeURIComponent(url)}`, null, {
                        headers: {
                            'Content-Type': 'application/json',
                        },
                    })
                );
            }

            // Handle file upload if files are present
            if (pdfFiles.length > 0) {
                const formData = new FormData();
                pdfFiles.forEach(file => {
                    formData.append('pdfFiles', file);
                });
                
                requests.push(
                    axios.post(`${config.API_BASE_URL}${config.ENDPOINTS.UPLOAD}`, formData, {
                        headers: {
                            'Content-Type': 'multipart/form-data',
                        },
                    })
                );
            }

            // Execute all requests simultaneously
            const responses = await Promise.all(requests);
            
            setSuccess(true);
            setUrl('');
            setPdfFiles([]);
            
            // Call success callback with the responses
            onSuccess(responses);
        } catch (error) {
            const errorMessage = error.response?.data?.message || error.message || 'Error processing your request. Please try again.';
            setError(errorMessage);
            onError?.(errorMessage);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="file-upload-container">
            <form onSubmit={handleSubmit} className="upload-form">
                <div className="form-group">
                    <label htmlFor="url">URL:</label>
                    <input
                        id="url"
                        type="url"
                        value={url}
                        onChange={(e) => {
                            setUrl(e.target.value);
                            setError(null);
                        }}
                        placeholder="Enter URL (optional)"
                        disabled={isLoading}
                    />
                </div>
                
                <div className="form-group">
                    <label htmlFor="pdfFiles">Upload PDF files:</label>
                    <input
                        id="pdfFiles"
                        type="file"
                        accept="application/pdf"
                        multiple
                        onChange={handleFileChange}
                        disabled={isLoading}
                    />
                    {pdfFiles.length > 0 && (
                        <div className="file-list">
                            Selected files: {pdfFiles.map(file => file.name).join(', ')}
                        </div>
                    )}
                </div>

                {error && <div className="error-message">{error}</div>}
                {success && <div className="success-message">Processing complete! Redirecting...</div>}

                <button 
                    type="submit" 
                    className="submit-button"
                    disabled={isLoading}
                >
                    {isLoading ? 'Processing...' : 'Submit'}
                </button>
            </form>
        </div>
    );
};

export default FileUpload;