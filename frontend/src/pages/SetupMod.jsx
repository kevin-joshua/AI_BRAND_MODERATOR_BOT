import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import FileUpload from '../components/FileUpload';
import './SetupMod.css';

  const SetupMod = () => {
    const [error, setError] = useState(null);

    

    return (
        <div className="setup-mod-container">
            <div className="setup-header">
                <h1>AI Module Setup</h1>
                <p className="description">
                    Upload your PDF documents or provide a URL to get started. Once processed, you'll be able to query your documents.
                </p>
            </div>

            <div className="setup-content">
                <div className="tab-content">
                    <h2>Upload Documents</h2>
                    <p className="description">
                        Upload PDF documents or provide a URL to process. These documents will be used for querying.
                    </p>
                    <FileUpload onError={setError} />
                    {error && <div className="error-message">{error}</div>}
                </div>
            </div>
        </div>
    );
};

export default SetupMod; 