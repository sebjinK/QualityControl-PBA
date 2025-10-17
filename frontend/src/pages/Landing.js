import React, { useState, useEffect } from "react";
import { Link, useNavigate } from 'react-router-dom';
import axios from "axios";
import Mission from './Mission';
import Caliber from './Caliber';

{ /*
  Filename:    Landing.js
  Description: This page is the landing page for the Bridging Hope website. It contains basic information about the website, its features, and its vision.
*/ }


const Landing = () => {
  let newDate = new Date();
  let year = newDate.getFullYear();
  const [signInNavigator, setSignInNavigator] = React.useState("/signin");
  const [activeSection, setActiveSection] = useState('mission'); // Default to Mission

  useEffect(() => {
    async function checkSession() {
      try {
        let url = process.env.REACT_APP_BACKURL;

        const response = await axios.get(url + "/api/dashboard", {
          withCredentials: true, // Send cookies
        });

        if (response.status !== 200) {
          //bad session id sign in direct to sign in page
          setSignInNavigator("/signin");
        }
        else {
          // If the response is successful, you can handle the data here
          setSignInNavigator("/dashboard");
        }
      } catch (error) {
        setSignInNavigator("/signin");
      }
    }
    checkSession();
  }, []);

  return (
    <React.Fragment>
      <div className="col-12 flex-grow-1">
        {/* navbar */}
        <nav className="nav navbar navbar-expand-xl navbar-light iq-navbar">
          <div className="container-fluid navbar-inner">
            <a href="/" className="navbar-brand">
              <div className="logo-main">
                <div className="logo-normal">
                  <img src='./images/logo_portobello_america.png'
                    className="img-fluid" alt="logo" style={{ maxHeight: "45px" }} />
                </div>
                <div className="logo-mini">
                  <svg className="text-primary icon-30" viewBox="0 0 30 30" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="-0.757324" y="19.2427" width="28" height="4" rx="2" transform="rotate(-45 -0.757324 19.2427)" fill="currentColor"></rect>
                    <rect x="7.72803" y="27.728" width="28" height="4" rx="2" transform="rotate(-45 7.72803 27.728)" fill="currentColor"></rect>
                    <rect x="10.5366" y="16.3945" width="16" height="4" rx="2" transform="rotate(45 10.5366 16.3945)" fill="currentColor"></rect>
                    <rect x="10.5562" y="-0.556152" width="28" height="4" rx="2" transform="rotate(45 10.5562 -0.556152)" fill="currentColor"></rect>
                  </svg>
                </div>
              </div>
              <h4 className="logo-title ps-2">Quality Control</h4>
              <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                <span className="navbar-toggler-icon"></span>
              </button>
              <div className="collapse navbar-collapse" id="navbarSupportedContent">
                <ul className="navbar-nav me-auto mb-2 mb-lg-0">
                  <li className="nav-item">
                    <a
                      className={`nav-link ms-4 ${activeSection === 'mission' ? 'active' : ''}`}
                      aria-current="page"
                      href="#"
                      onClick={() => setActiveSection('mission')}
                    >
                      Mission
                    </a>
                  </li>
                  <li className="nav-item">
                    <a
                      className={`nav-link ms-4 ${activeSection === 'caliber' ? 'active' : ''}`}
                      href="#"
                      onClick={() => setActiveSection('caliber')}
                    >
                      Caliber Analysis
                    </a>
                  </li>
                </ul>
              </div>
            </a>

            <div className="d-flex justify-content-around align-items-center">
              <Link to={signInNavigator} className="btn btn-primary btn-sm align-items-center me-2">Sign In</Link>
              <Link to="/register" className="btn btn-secondary btn-sm align-items-center">Register</Link>
            </div>
          </div>
        </nav>

        {/* background image and info */}
        <div className="col-12">
          <div className="h-75" style={{ backgroundImage: `url(./images/landing-hero.jpeg)`, backgroundRepeat: "no-repeat", backgroundAttachment: "fixed", backgroundSize: "cover" }}>
            {/* attribution: https://www.istockphoto.com/photo/happy-community-service-volunteers-cleaning-up-the-park-gathered-around-stacking-gm1410242950-460512007 */}
            <div className="row justify-content-end align-items-center" style={{ minHeight: "100%", important: false }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
                className="col-12 col-md-6 d-flex"
              >
                <img
                  src='./images/logo_portobello_america.png'
                  className="img-fluid" // ADDED: rounded-3 for medium-sized rounded corners
                  alt="logo"
                  style={{
                    maxHeight: "156px",
                    backgroundColor: "rgba(255, 255, 255, 0.6)",
                    display: 'block',
                    margin: '0 auto',
                    position: 'relative',
                  }}
                />
              </div>
              <div className="col-12 col-md-6" style={{ backgroundColor: 'rgba(180,180,180,.40)', minHeight: "100%" }}>
                <h2 className="text-white mt-4 text-center">Quality Control for Tile Manufacturing using AI</h2>
                <div className="row col-12 justify-content-center my-4">
                  <Link to="/clienthelp" className="btn btn-primary col-6 mb-1">Documentation</Link>
                </div>
              </div>
            </div>
          </div>

          <div className="flex-grow-1">
            <div id="content-section">
              {activeSection === 'mission' && <Mission />}
              {activeSection === 'caliber' && <Caliber />}
            </div>
          </div>

        </div>
      </div>
      {/* footer */}
      <footer className="footer">
        <div className="footer-body d-flex justify-content-between mx-4 pb-3">
          <ul className="list-inline mb-0 p-0">
            <li className="list-inline-item"><a href="./dashboard/extra/privacy-policy.html">Privacy Policy</a></li>
            <li className="list-inline-item"><a href="./dashboard/extra/terms-of-service.html">Terms of Use</a></li>
          </ul>
          <div className="right-panel">
            <p>© {year} Bridging Hope. All Rights Reserved.</p>
          </div>
        </div>
      </footer>
    </React.Fragment>
  );
}

export default Landing;
