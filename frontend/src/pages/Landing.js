import React, { useState, useEffect } from "react";
import { Link, useNavigate } from 'react-router-dom';
import axios from "axios";

{ /*
  Filename:    Landing.js
  Description: This page is the landing page for the Bridging Hope website. It contains basic information about the website, its features, and its vision.
*/ }


const Landing = () => {
  let newDate = new Date();
  let year = newDate.getFullYear();
  const [signInNavigator, setSignInNavigator] = React.useState("/signin");

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
            <span className="caliber-link ps-3 color text-muted">Caliber</span>
            <span className="labels-link ps-3 color text-muted">Label Verification</span>
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

        {/* Our Vision */}
        <div id="vision" className="col-12 row mb-5 justify-content-center">
          <div className="col-12 col-md-6 ">
            <div className="">
              <h2 className="text-center col-12 mt-5">Mission</h2>
              <hr />
              <p className="text-center col-12 mt-3">Our vision is to create a world where non-profits can easily connect and share resources to better serve their communities. We believe that by providing a platform for non-profits to connect and share resources, we can help them better serve their communities and make a positive impact on the world.</p>
            </div>
          </div>
        </div>

        {/* More info cards */}
        <div className="col-12 row mt-4 mb-5 justify-content-around">
          <div className="col-12 col-md-3 card">
            <div className="card-body">
              <h2 className="text-center col-12 mt-2">Areas Served</h2>
              <hr />
              <p className="col-12 mt-4"><i className="bi bi-heart-fill me-2 text-info"></i>Algood</p>
              <p className="col-12 mt-2"><i className="bi bi-heart-fill me-2 text-info"></i>Baxter</p>
              <p className="col-12 mt-2"><i className="bi bi-heart-fill me-2 text-info"></i>Cookeville</p>
              <p className="col-12 mt-2"><i className="bi bi-heart-fill me-2 text-info"></i>Crossville</p>
              <p className="col-12 mt-2"><i className="bi bi-heart-fill me-2 text-info"></i>Cumberland County</p>
              <p className="col-12 mt-2"><i className="bi bi-heart-fill me-2 text-info"></i>Monterey</p>
              <p className="col-12 mt-2"><i className="bi bi-heart-fill me-2 text-info"></i>Putnam County</p>
            </div>
          </div>
          <div className="col-12 col-md-3 card">
            <div className="card-body d-flex flex-column">
              <div className="flex-grow-1">
                <h2 className="text-center col-12 mt-2">Partners</h2>
                <hr />
                <p className="col-12 mt-4"><i className="bi bi-heart-fill me-2 text-info"></i>DUO Mobile Missions</p>
                <p className="col-12 mt-2"><i className="bi bi-heart-fill me-2 text-info"></i>Life Church</p>
                <p className="col-12 mt-2"><i className="bi bi-heart-fill me-2 text-info"></i>Putnam County</p>
                <p className="col-12 mt-2"><i className="bi bi-heart-fill me-2 text-info"></i>Steven's Street Baptist Church</p>
                <p className="col-12 mt-2"><i className="bi bi-heart-fill me-2 text-info"></i>Upper Cumberland Family Justice Center</p>
              </div>
              <Link to="/404" className="btn btn-secondary col-12">Contact Us to Add Your Organization!</Link> {/* TODO make this link to an email */}
            </div>
          </div>
          <div className="col-12 col-md-3 card">
            <div className="card-body d-flex flex-column">
              <div className="flex-grow-1">
                <h2 className="text-center col-12 mt-2">Features</h2>
                <hr />
                <p className="col-12 mt-4 mb-0"><i className="bi bi-heart-fill me-2 text-info"></i>Register Clients</p>
                <p className="col-12 mt-0 ms-4"><small className="fw-lighter fst-italic">Add new clients to Bridging Hope</small></p>
                <p className="col-12 mt-2 mb-0"><i className="bi bi-heart-fill me-2 text-info"></i>Log Visits</p>
                <p className="col-12 mt-0 ms-4"><small className="fw-lighter fst-italic">Track the aid you give digitally</small></p>
                <p className="col-12 mt-2 mb-0"><i className="bi bi-heart-fill me-2 text-info"></i>Search Clients</p>
                <p className="col-12 mt-0 ms-4"><small className="fw-lighter fst-italic">Find existing client accounts with ease</small></p>
              </div>
              <Link to="/faq" className="btn btn-secondary col-12">Frequently Asked Questions</Link>
            </div>
          </div>
        </div>

        {/* Register Now button */}
        <div className="col-12 row mt-4 mb-5 justify-content-around">
          <Link to="/register" className="btn btn-secondary btn-lg col-8">Register Now</Link>
        </div>

        {/* Developed and maintained with love */}
        <div className="col-12 row mb-5 justify-content-center">
          <p className="text-center">Developed and maintained with<i className="bi bi-heart-fill ms-2 me-2 text-info"></i>by Business Information Technology and Computer Science students at Tennessee Technological University</p>
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
      </div>
    </React.Fragment>
  );
}

export default Landing;
