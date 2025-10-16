import React from 'react';
// Assuming 'Link' is imported from 'react-router-dom' based on its usage
import { Link } from 'react-router-dom';

const Mission = () => {
  return (
    <div className="container">
      {/* Our Vision / Mission Section (Centered) */}
      <div id="vision" className="col-12 row mb-5 justify-content-center">
        <div className="col-12 col-md-6 ">
          <h2 className="text-center col-12 mt-5">Caliber</h2>
          <hr />
          <p className="text-gray-600 leading-relaxed">
            We're building two powerful AI quality control tools to completely eliminate human error in tile manufacturing, bringing in automated efficiency and standardized consistency.
          </p>

          {/* Calibre Assessment */}
          <h5 className="font-semibold text-gray-800 mb-2">Calibre Assessment</h5>
          <ul className="list-disc ml-5 text-gray-600 space-y-3">
            <li>
              <ul className="list-circle ml-5">
                <li>We're training a Convolutional Neural Network (CNN) to visually inspect every finished tile.</li>
                <li>The system determines the tile's exact calibre (categorized 3, 4, or 5).</li>
              </ul>
            </li>
          </ul>

          {/* Automated Labeling */}
          <h5 className="font-semibold text-gray-800 mb-2">Automated Labeling</h5>
          <ul className="list-disc ml-5 text-gray-600 space-y-3">
            <li>
              <ul className="list-circle ml-5">
                <li>We use computer vision to quickly read the existing product ID printed on the box.</li>
                <li>A new, correct label is automatically printed with all the shipment details.</li>
              </ul>
            </li>
          </ul>

        </div>
      </div>

      {/* More info cards (Areas Served, Partners, Features) */}
      <div className="col-12 row mt-4 mb-5 justify-content-around">

        {/* Areas Served Card */}
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

        {/* Partners Card */}
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
            {/* The link target needs to be a valid path/function */}
            <Link to="/404" className="btn btn-secondary col-12">Contact Us to Add Your Organization!</Link>
          </div>
        </div>

        {/* Features Card */}
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
    </div>
  );
};

export default Mission;
