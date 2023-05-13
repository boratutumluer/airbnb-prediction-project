mapboxgl.accessToken = APP_VAR.access_token;

        const map = new mapboxgl.Map({
            container: 'map',
            style: 'mapbox://styles/boratutumluer/clh0i109u00dl01qtcq4mee7s',
            center: [29.02573, 41.08384],
            maxBounds: [[27.965767, 40.801789],[29.963703, 41.585417]],
            zoom: 12
        });

        map.fitBounds([[27.965767, 40.801789],[29.963703, 41.585417]])

        function highlightNeighbourhood(district) {
          if (map.getSource("neighbourhood_highlight_source")) {
            map.setLayoutProperty("neighbourhood_highlight", "visibility", "visible");
            map
              .getSource("neighbourhood_highlight_source")
              .setData(
                neighbour_geojson.features.filter(
                  (i) => i.properties.neighbourhood === district
                )[0]
              );
          } else {
            map.addSource("neighbourhood_highlight_source", {
              type: "geojson",
              data: neighbour_geojson.features[0],
            });
            map.addLayer({
              id: "neighbourhood_highlight",
              type: "line",
              source: "neighbourhood_highlight_source",
              paint: {
                "line-color": "white",
                "line-opacity": 0.75,
                "line-width": 2,
              },
            });
            map.setLayoutProperty("neighbourhood_highlight", "visibility", "none");
          }
        }

        function updateRoomChartData(){
            roomTypeChart.data.datasets[0].data = [14765, 194, 4563, 2000];
            roomTypeChart.update()
        }

        function updateCapacityChartData(){
            capacityChart.data.datasets[0].data = [14765, 194, 4563, 2000];
            capacityChart.update();
        }


// #####################################################################################################################
                                                    <!--NEIGHBOURHOOD-->
        const neighbour_geojson =  JSON.parse(APP_VAR.neighbourhood_json);
        map.on('load', function() {
            map.addSource('neighbourhoods', {
                type: 'geojson',
                data: neighbour_geojson
            });
            map.addLayer({
                id: 'neighbourhoods-layer',
                type: 'fill',
                source: 'neighbourhoods',
                paint: {
                    'fill-color': 'white',
                    'fill-opacity': 0.2
                },
            });
            map.addLayer({
                id: "neighbourhoods-label",
                type: "symbol",
                source: "neighbourhoods",
                layout: {
                "text-field": ["get", "neighbourhood"],
                "text-variable-anchor": ["top", "bottom", "left", "right"],
                "text-radial-offset": 0.5,
                "text-justify": "auto",
                "icon-image": ["concat", ["get", "icon"], "-15"],
                "text-size": 10,
                }
            });
            // highlight layer is initialized!!
            highlightNeighbourhood('');
        });

// #####################################################################################################################
                                                    <!--POINTS-->
        const Points = APP_VAR.points;
        map.on('load', function() {
            map.addSource('point', {
              'type': 'geojson',
              'data': {
                'type': 'FeatureCollection',
                'features': Points.map(function(point) {
                  return {
                    'type': 'Feature',
                    'properties' : {'label' : point[1], 'segment': point[2],
                                    'description' : '<strong>' + point[4] + '</strong>' +
                                    '<a href="' + point[3] + '" target="_blank" </a>'},
                    'geometry': {
                      'type': 'Point',
                      'coordinates': point[0]
                    }
                  };
                })
              },
                cluster: true,
                clusterMaxZoom: 14,
                clusterRadius: 50
            });

            map.addLayer({
                id: 'clusters',
                type: 'circle',
                source: 'point',
                filter: ['has', 'point_count'],
                paint: {
                    'circle-color': [
                        'step',
                        ['get', 'point_count'],
                        '#146C94',
                        100,
                        '#EBB02D',
                        750,
                        '#E74646'
                    ],
                    'circle-radius': [
                        'step',
                        ['get', 'point_count'],
                        20,
                        100,
                        30,
                        750,
                        40
                    ]
                }
            });

            map.addLayer({
                id: 'cluster-count',
                type: 'symbol',
                source: 'point',
                filter: ['has', 'point_count'],
                layout: {
                    'text-field': ['get', 'point_count_abbreviated'],
                    'text-font': ['DIN Offc Pro Medium', 'Arial Unicode MS Bold'],
                    'text-size': 12
                }
            });

            map.addLayer({
                id: 'unclustered-point',
                type: 'circle',
                source: 'point',
                filter: ['!', ['has', 'point_count']],
                paint: {
                    'circle-color': '#11b4da',
                    'circle-radius': 3,
                    'circle-stroke-width': 1,
                    'circle-stroke-color': 'black'
                }
            });

            map.addSource('heat_points', {
              'type': 'geojson',
              'data': {
                'type': 'FeatureCollection',
                'features': Points.map(function(point) {
                    return {
                        'type': 'Feature',
                        'properties' : {'label' : point[1], 'segment': point[2]},
                        'geometry': {
                        'type': 'Point',
                        'coordinates': point[0]
                        }
                    };
                })
              },
            });


            map.addLayer({
              id: "heatmap_layer",
              type: "heatmap",
              source: "heat_points",
              paint: {
                // Increase the heatmap weight based on frequency and property magnitude
                "heatmap-weight": ["interpolate", ["linear"], ["get", "mag"], 0, 0, 6, 1],
                // Increase the heatmap color weight weight by zoom level
                // heatmap-intensity is a multiplier on top of heatmap-weight
                "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 0, 1, 19, 3],
                // Color ramp for heatmap.  Domain is 0 (low) to 1 (high).
                // Begin color ramp at 0-stop with a 0-transparancy color
                // to create a blur-like effect.
                "heatmap-color": [
                  "interpolate",
                  ["linear"],
                  ["heatmap-density"],
                  0,
                  "rgba(33,102,172,0)",
                  0.2,
                  "rgb(103,169,207)",
                  0.4,
                  "rgb(209,229,240)",
                  0.6,
                  "rgb(253,219,199)",
                  0.8,
                  "rgb(239,138,98)",
                  1,
                  "rgb(178,24,43)",
                ],
                // Adjust the heatmap radius by zoom level
                "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 0, 2, 19, 20],
                // Transition from heatmap to circle layer by zoom level
                "heatmap-opacity": ["interpolate", ["linear"], ["zoom"], 7, 1, 19, 0],
              },
            });
            map.setLayoutProperty('heatmap_layer','visibility','none');

        });


// #####################################################################################################################
                                        <!--        NEIGHBOURHOOD FILTER-->
        const neighbourhoods_bbox = APP_VAR.bbox;
        document.getElementById('select-district').addEventListener('change', (event) => {

            map.setLayoutProperty('unclustered-point','visibility','visible');
            const district = event.target.value;
            map.setFilter('unclustered-point', ['==', ['get', 'label'], district]);


            map.setPaintProperty('unclustered-point', 'circle-color', '#11b4da');
            map.setPaintProperty('unclustered-point', 'circle-radius', 3);
            map.setPaintProperty('unclustered-point', 'circle-stroke-width', 1);
            map.setPaintProperty('unclustered-point', 'circle-stroke-color', 'black');


            highlightNeighbourhood(district);
            const district_bbox = neighbourhoods_bbox[district];
            map.fitBounds(district_bbox, {
                maxZoom: 14,
                duration: 2000
            });

            const style = map.getStyle()
            style.sources.point.cluster = false
            map.setStyle(style)
        });

// #####################################################################################################################
                                                    <!-- HEATMAP -->
        const heatmapButton = document.querySelector('#heatmapBtn');
        heatmapButton.addEventListener('click',(event)=>{
            if(event.target.innerText === 'Heatmap'){
                event.target.innerText = 'Point Cloud';
                map.setLayoutProperty('heatmap_layer','visibility','visible');
                map.setLayoutProperty('unclustered-point','visibility','none');
                map.setLayoutProperty('clusters','visibility','none');
                map.setLayoutProperty('cluster-count','visibility','none');

            } else {
                event.target.innerText = 'Heatmap';
                map.setLayoutProperty('heatmap_layer','visibility','none');
                map.setLayoutProperty('clusters','visibility','visible');
                map.setLayoutProperty('cluster-count','visibility','visible');
                map.setLayoutProperty('unclustered-point','visibility','visible');
                map.setLayoutProperty('all-points','visibility','none');
                const style = map.getStyle()
                style.sources.point.cluster = true
                map.setStyle(style)
            }
        })

// #####################################################################################################################
                                                    <!--CLUSTER -->
        const clusterButton = document.querySelector('#clusterBtn');
        clusterButton.addEventListener('click',(event)=>{
                map.setLayoutProperty('heatmap_layer','visibility','none');
                map.setLayoutProperty('unclustered-point','visibility','visible');
                map.setLayoutProperty('clusters','visibility','none');
                map.setLayoutProperty('cluster-count','visibility','none');

                map.setPaintProperty('unclustered-point', 'circle-color', {
                    "property": "segment",
                    "type": "categorical",
                    "stops": [
                      [1, "red"],
                      [2, "green"],
                      [3, "blue"],
                      [4, "yellow"],
                      [5, "purple"],
                      [6, "orange"],
                      [7, "white"],
                      [8, "brown"]
                      ],
                    "default": "gray"
                });
        });

// #####################################################################################################################
                                                    <!-- LIMIT -->
        document.getElementById('set_bbox').addEventListener('click', () => {
            map.fitBounds([[27.965767, 40.801789],[29.963703, 41.585417]])
            const style = map.getStyle()
            style.sources.point.cluster = true
            map.setStyle(style)

            map.setFilter('unclustered-point', null)

            map.setLayoutProperty('neighbourhood_highlight','visibility','none');
            map.setLayoutProperty('all-points','visibility','none');
            map.setLayoutProperty('clusters','visibility','visible');
            map.setLayoutProperty('cluster-count','visibility','visible');
            map.setLayoutProperty('unclustered-point','visibility','none');

        });
// #####################################################################################################################
                                                    <!--POP-UP-->

        map.on('click', 'unclustered-point', (k) => {
            const coordinates = k.features[0].geometry.coordinates.slice();
            const description = k.features[0].properties.description;

            while (Math.abs(k.lngLat.lng - coordinates[0]) > 180) {
                coordinates[0] += k.lngLat.lng > coordinates[0] ? 360 : -360;
            }

            new mapboxgl.Popup()
                .setLngLat(coordinates)
                .setHTML(description)
                .addTo(map);
        });
        map.on('mouseenter', 'unclustered-point', () => {map.getCanvas().style.cursor = 'pointer';});
        map.on('mouseleave', 'unclustered-point', () => {map.getCanvas().style.cursor = '';});

// #####################################################################################################################
                                                <!--ADD NEW POINT-->

        map.on('click', function(e) {
			var lngLat = e.lngLat;
			var popup = new mapboxgl.Popup()
				.setLngLat(lngLat)
				.setHTML('<form id="popup-form"><label>Property name:</label><br><input type="text" id="property-name"><br><label>Property value:</label><br><input type="text" id="property-value"><br><button type="submit">Save</button></form>')
				.addTo(map);

			document.getElementById('popup-form').addEventListener('submit', function(e) {
				e.preventDefault();
				var propertyName = document.getElementById('property-name').value;
				var propertyValue = document.getElementById('property-value').value;
<!--				saveProperty(lngLat, propertyName, propertyValue);-->
				popup.remove();
			});
		});

<!--		function saveProperty(lngLat, name, value) {-->
<!--			var url = 'YOUR_API_URL'; // Your API endpoint URL-->
<!--			var data = { lng: lngLat.lng, lat: lngLat.lat, name: name, value: value };-->
<!--			fetch(url, {-->
<!--				method: 'POST',-->
<!--				body: JSON.stringify(data),-->
<!--				headers: {-->
<!--					'Content-Type': 'application/json'-->
<!--				}-->
<!--			})-->
<!--			.then(response => response.json())-->
<!--			.then(data => console.log(data))-->
<!--			.catch(error => console.error(error));-->
<!--		}-->













// #####################################################################################################################

                                        <!--        STATISTICS-->
        var accommodates_price = document.getElementById('accommodates_price_chart').getContext('2d');
        const capacityChart = new Chart(accommodates_price, {
            type: "line",
            data: {
                labels: APP_VAR.accommodations ,
                datasets: [{
                    label: 'capacity',
                    backgroundColor: "rgba(0,0,255,1.0)",
                    borderColor: "rgba(0,0,255,0.1)",
                    data: APP_VAR.prices,
                    fill: false
                }]
            },
            options: {
                title: {
                    display: true,
                    text: "Prices by Capacity"
                },
                scales: {
                    yAxes: [{
                        ticks: {
                            beginAtZero: true
                        },
                        scaleLabel: {
                            display: true,
                            labelString: "Price in TL"
                        }
                    }],
                    xAxes: [{
                        scaleLabel: {
                            display: true,
                            labelString: "Capacity"
                        }
                    }]
                },
                  legend: {
                     display: false
                        }
            }
        });

       var room_type_ = document.getElementById('room_type_chart').getContext('2d');
         const roomTypeChart = new Chart(room_type_, {
            type: "horizontalBar",
            data: {
                labels: ['Entire home/apt', 'Private room', 'Hotel room', 'Shared room'],
                datasets: [{
                    backgroundColor: ["rgb(0, 35, 91)", "rgb(226, 24, 24)", "rgb(255, 221, 131)", "rgb(152, 223, 214)"],
                    borderColor: "rgba(0,0,255,0.1)",
                    data: [14765, 4563, 194, 100],
                    fill: false
                }]
            },
            options: {
                title: {
                    display: true,
                    text: "Room Type"
                },
                scales: {
                    yAxes: [{
                        ticks: {
                            beginAtZero: true
                        },
                    }],
                    xAxes: [{
                        scaleLabel: {
                            display: true,
                            labelString: "Count"
                        }
                    }]
                },
                legend: {
                     display: false
                        }
            }
        });