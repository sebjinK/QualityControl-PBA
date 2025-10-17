import React from 'react';
// Assuming 'Link' is imported from 'react-router-dom' based on its usage
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div className="container">
      {/* Our Vision / Home Section (Centered) */}
      <div id="vision" className="col-12 row mb-3 justify-content-center">
        <div className="col-12 col-md-6 ">
          <h2 className="text-center col-12 mt-5"><span class="opacity-50 pe-2">🙘</span> Quality Control AI Toolset <span class="opacity-50 ps-2">🙚</span></h2>
          <hr />
          <p className="text-gray-600 leading-relaxed" style={{ textAlign: 'justify' }}>
            We are building two powerful AI quality control tools to completely eliminate human error in tile manufacturing, bringing in automated efficiency and standardized consistency.
          </p>

          {/* Calibre Assessment */}
          <h5 className="font-semibold text-gray-800 mb-2">Calibre Analysis</h5>
          <ul className="list-circle ml-5">
            <li>We trained a Convolutional Neural Network (CNN) to inspect finished tiles.</li>
            <li>The system determines the tile's exact calibre in mm (categorized 3, 4, or 5).</li>
          </ul>

          {/* Automated Labeling */}
          <h5 className="font-semibold text-gray-800 mb-2">Label Automation</h5>
          <ul className="list-circle ml-5">
            <li>We use a CNNs with OCR to quickly read the product ID printed on the box.</li>
            <li>A new, correct label will be printed with all the shipment details.</li>
            <li>Box labels can be put against printed labels to verify they are the same.</li>
          </ul>

        </div>
      </div>

      {/* More info cards (Areas Served, Partners, Features) */}
      <div className="col-12 row mt-4 gap-5 justify-content-center">

        {/* Partners Card */}
        <div className="col-12 col-md-3 card">
          <div className="card-body d-flex flex-column">
            <div className="flex-grow-1">
              <h3 className="text-center col-12 mt-2">The Team</h3>
              <hr />
              <p className="col-12 mt-4"><i className="bi bi-heart-fill me-2 text-info"></i>Jamie Boyd</p>
              <p className="col-12 mt-2"><i className="bi bi-heart-fill me-2 text-info"></i>Braylon Trail</p>
              <p className="col-12 mt-2"><i className="bi bi-heart-fill me-2 text-info"></i>Guilermme Gripp</p>
              <p className="col-12 mt-2"><i className="bi bi-heart-fill me-2 text-info"></i>Preston Harberts</p>
              <p className="col-12 mt-2 pb-3"><i className="bi bi-heart-fill me-2 text-info"></i>Sebjin Kennedy</p>
            </div>
            {/* The link target needs to be a valid path/function */}
            <Link to="/404" className="btn btn-secondary col-12">Contact Us</Link>
          </div>
        </div>

        {/* Features Card */}
        <div className="col-12 col-md-3 card">
          <div className="card-body d-flex flex-column">
            <div className="flex-grow-1">
              <h3 className="text-center col-12 mt-2">Features</h3>
              <hr />
              <p className="col-12 mt-4 mb-0"><i className="bi bi-heart-fill me-2 text-info"></i>Caliber Analysis</p>
              <p className="col-12 mt-0 ms-4"><small className="fw-lighter fst-italic">Measuring Size of Tiles through Production</small></p>
              <p className="col-12 mt-2 mb-0"><i className="bi bi-heart-fill me-2 text-info"></i>Label Automation</p>
              <p className="col-12 mt-0 ms-4"><small className="fw-lighter fst-italic">Verification and packaging of Labels</small></p>
            </div>
            <Link to="/faq" className="btn btn-secondary col-12">FAQ</Link>
          </div>
        </div>
      </div>

      {/* Developed and maintained with love */}
      <div className="col-12 row mb-3 justify-content-center">
        <p className="text-center">Developed by Team 1 (Quality Control)</p>
      </div>
    </div>
  );
};

export default Home;