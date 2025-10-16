//Filename: ClientMap.tsx
//Purpose: This file contains the code for the ClientMap component.
//         It is a map of locations where clients can get aid.

import React, { useEffect, useCallback, useState } from "react";

//Google Maps imports
import {
  APIProvider,
  Map,
  useMap,
  useMapsLibrary,
  AdvancedMarker,
  Pin,
  InfoWindow,
  useAdvancedMarkerRef,
  AdvancedMarkerProps,
  AdvancedMarkerAnchorPoint,
} from "@vis.gl/react-google-maps";

const data = getData()
  /* .sort((a, b) => b.position.lat - a.position.lat) */ // Sort by latitude (this messes up the array, but I am keeping the code here for reference)
  .map((dataItem, index) => ({ ...dataItem, zIndex: index }));

const Z_INDEX_SELECTED = data.length;
const Z_INDEX_HOVER = data.length + 1;

const ClientMap = () => {
  const apiKey = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;

  if (!apiKey) {
    throw new Error('REACT_APP_GOOGLE_MAPS_API_KEY is required.');
  }

  const [markers] = useState(data);

  const [hoverId, setHoverId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [selectedMarker, setSelectedMarker] =
    useState<google.maps.marker.AdvancedMarkerElement | null>(null);
  const [infoWindowShown, setInfoWindowShown] = useState(false);

  const onMouseEnter = useCallback((id: string | null) => setHoverId(id), []);
  const onMouseLeave = useCallback(() => setHoverId(null), []);
  const onMarkerClick = useCallback(
    (id: string | null, marker?: google.maps.marker.AdvancedMarkerElement) => {
      setSelectedId(id);

      if (marker) {
        setSelectedMarker(marker);
      }

      if (id !== selectedId) {
        setInfoWindowShown(true);
      } else {
        setInfoWindowShown((isShown) => !isShown);
      }
    },
    [selectedId]
  );

  const onMapClick = useCallback(() => {
    setSelectedId(null);
    setSelectedMarker(null);
    setInfoWindowShown(false);
  }, []);

  const handleInfowindowCloseClick = useCallback(
    () => setInfoWindowShown(false),
    []
  );


  return(
    <APIProvider apiKey={apiKey} onLoad={() => console.log('Maps API has loaded.')} >
    {/* <APIProvider apiKey={process.env.REACT_APP_GOOGLE_MAPS_API_KEY} onLoad={() => console.log('Maps API has loaded.')}> */}

      <Map
        defaultZoom={12}
        defaultCenter={{ lat: 36.162839, lng: -85.5016423 }}
        mapId="44349268b190049a"
        onClick={onMapClick}
        clickableIcons={false}
        disableDefaultUI
      >
        {markers.map(({ id, zIndex: zIndexDefault, position }) => {
          let zIndex = zIndexDefault;

          if (hoverId === id) {
            zIndex = Z_INDEX_HOVER;
          }

          if (selectedId === id) {
            zIndex = Z_INDEX_SELECTED;
          }

          return (
            <AdvancedMarkerWithRef
              onMarkerClick={(
                marker: google.maps.marker.AdvancedMarkerElement
              ) => onMarkerClick(id, marker)}
              onMouseEnter={() => onMouseEnter(id)}
              onMouseLeave={onMouseLeave}
              key={id}
              zIndex={zIndex}
              className="custom-marker"
              style={{
                transform: `scale(${
                  [hoverId, selectedId].includes(id) ? 1.3 : 1
                })`,
                transformOrigin: AdvancedMarkerAnchorPoint["BOTTOM"].join(" "),
              }}
              position={position}
            >
              <Pin
                background={selectedId === id ? "#22ccff" : null}
                borderColor={selectedId === id ? "#1e89a1" : null}
                glyphColor={selectedId === id ? "#0f677a" : null}
              />
            </AdvancedMarkerWithRef>
          );
        })}

        {infoWindowShown && selectedMarker && (
          <InfoWindow
            anchor={selectedMarker}
            pixelOffset={[0, -2]}
            onCloseClick={handleInfowindowCloseClick}
          >
            <h2>{markers[Number(selectedId)].name}</h2>
            <a
              href={`https://google.com/maps/search/${
                markers[Number(selectedId)].name
              }`}
            >
              See on Google Maps
            </a>
            <p>{markers[Number(selectedId)].description}</p>
          </InfoWindow>
        )}
      </Map>
    </APIProvider>
  );
};

const AdvancedMarkerWithRef = (
  props: AdvancedMarkerProps & {
    onMarkerClick: (marker: google.maps.marker.AdvancedMarkerElement) => void;
  }
) => {
  const { children, onMarkerClick, ...advancedMarkerProps } = props;
  const [markerRef, marker] = useAdvancedMarkerRef();

  return (
    <AdvancedMarker
      onClick={() => {
        if (marker) {
          onMarkerClick(marker);
        }
      }}
      ref={markerRef}
      {...advancedMarkerProps}
    >
      {children}
    </AdvancedMarker>
  );
};

// This type is used to define the data for the markers on the map
type MarkerData = Array<{
  id: string;
  name: string;
  description: string;
  address: string;
  position: google.maps.LatLngLiteral;
  zIndex: number;
}>;

/* function: getData
   purpose:  This function returns an array of locations where clients can get aid */
function getData() {
  const data: MarkerData = [
    {
      id: "0",
      name: "Cookeville Rescue Mission",
      description:
        "Cookeville Rescue Mission offers temporary emergency shelter to anyone who has need of it. " +
        "In addition to Emergency Shelter, we provide a long-term transitional program for men, women, " +
        "and families who are seeking to enrich their lives and break the cycle of homelessness. " +
        "Residents attend Life Recovery Groups to help them overcome the crushing weight of chemical dependence " +
        "and other life controlling habits. " +
        "Families and individuals receive food boxes that are prepared with donations received from local foodbanks, " +
        "and donations from individuals, churches, businesses, and civic groups.",
      address: "1331 S Jefferson Ave, Cookeville, TN 38506",
      position: { lat: 36.126104072519304, lng: -85.50700598196516 },
      zIndex: 0,
    },
    {
      id: "1",
      name: "Life Church",
      description:
        "Life Church is a place you can come to know the real Jesus, find good community," +
        "grow deeper in your faith, and help reach a world in need.",
      address: "2223 N Washington Ave, Cookeville, TN 38501",
      position: { lat: 36.19121147075565, lng: -85.49167830894544 },
      zIndex: 0,
    },
    {
      id: "2",
      name: "Cookeville First Baptist Church",
      description:
        "Cookeville First Baptist is made up of a group of believers who want to worship and do life together.",
      address: "18 S Walnut Ave, Cookeville, TN 38501",
      position: { lat: 36.1626903747245, lng: -85.50597237724672 },
      zIndex: 0,
    },
  ];

  return data;
}

export default ClientMap;
