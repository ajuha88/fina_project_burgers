var geojson = "http://data.beta.nyc//dataset/0ff93d2d-90ba-457c-9f7e-39e47bf2ac5f/resource/35dd04fb-81b3-479b-a074-a27a37888ce7/download/d085e2f8d0b54d4590b1e7d1f35594c1pediacitiesnycneighborhoods.geojson"

function setBins(inputData) {
    var sorted = inputData.sort(function(a, b){return a-b});
    var len = sorted.length;
    var per20 =  Math.floor(len*.2);

    var bins = [];
    for (i=0; i<5; i++) {
        bins.push(sorted[0 + i*per20]);
    }
    return bins
};

function chooseColor(bins, value) {
    var color;
    if (bins[0] <= value && value < bins[1]) {color = "red"}
    else if (bins[1] <= value && value < bins[2]) {color = "orange"}
    else if (bins[2] <= value && value < bins[3]) {color = "yellow"}
    else if (bins[3] <= value && value < bins[4]) {color = "blue"}
    else if (value >= bins[4]) {color = "green"};
    return color
};

function generateOverlay(selectedData) {
    var mapLayer = L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
        attribution: "Map data &copy; <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> contributors, <a href=\"https://creativecommons.org/licenses/by-sa/2.0/\">CC-BY-SA</a>, Imagery © <a href=\"https://www.mapbox.com/\">Mapbox</a>",
        maxZoom: 18,
        id: "mapbox.streets",
        accessToken: API_KEY
    });

    var layerGroup = L.layerGroup([mapLayer]);
    d3.csv("data/by_neighborhood.csv", function(csv) {
        var nhObject = {};
        var numberList = [];
        csv.forEach(function(row) {
            var unrounded = row[selectedData];
            // var rounded = Math.round(100*unrounded)/100;
            nhObject[row.Neighborhood] = unrounded;
            numberList.push(unrounded);
        });
        bins = setBins(numberList);
        d3.json(geojson, function(geo) {
            L.geoJson(geo, {
                style: function(feature) {
                    if (nhObject[feature.properties.neighborhood]) { return {
                        color: "white",
                        fillColor: chooseColor(bins, nhObject[feature.properties.neighborhood]),
                        fillOpacity: 0.5,
                        weight: .5
                    }}
                    else { return {
                        color: "None",
                    }}
                },
                onEachFeature: function(feature, layer) {
                    layer.on({
                    mouseover: function(event) {
                        layer = event.target;
                        layer.setStyle({fillOpacity: 0.8});
                    },
                    mouseout: function(event) {
                        layer = event.target;
                        layer.setStyle({fillOpacity: 0.5});
                    },
                    });
                    if (nhObject[feature.properties.neighborhood]) {
                        layer.bindPopup("<p>" + feature.properties.neighborhood + "</p> <p>" + 
                        selectedData + ": "  + "</p>" + 
                        "<p>" + nhObject[feature.properties.neighborhood] + "</p>");
                    } 
                }}
            ).addTo(layerGroup);
        });
        });
    return layerGroup;
};

var markerLayer = L.layerGroup([]);
d3.csv("data/merged_data.csv", function(csv) {
    csv.forEach(function(row) {
        var marker = L.marker([row['Venue Latitude'], row['Venue Longitude']]).addTo(markerLayer);
        marker.bindPopup(`${row['Venue']}: 
        <br>Rating - ${row['Rating']}
        <br>${row['Like Count']} Likes 
        <br>${row['Tip Count']} Reviews`)
    });
});

var baseMaps = {
    "By Number of Burger Places" : generateOverlay("Venue"),
    "By Average Rating" : generateOverlay("Rating"),
    "By Number of Likes" : generateOverlay("Like Count"),
    "By Number of Reviews" : generateOverlay("Tip Count"),
};

var overlayMaps = {
    "Venues": markerLayer
};

var map = L.map("map", {
    center: [40.77, -74],
    zoom: 11
});    

var legend = L.control({position: 'bottomright'});
legend.onAdd = function (map) {
    var div = L.DomUtil.create('div', 'info legend'),
    grades = [1, 2, 3, 4, 5],
    labels = ["Lower 20%", "20% to 40%", "40% to 60%", "60% to 80%", "Top 20%"];
    for (var i = 0; i < grades.length; i++) {
    div.innerHTML +=
        '<i style="background:' + chooseColor(grades, grades[i] + .5) + '"></i> ' +
        labels[i] + (labels[i + 1] ?'<br>' : '');
    };
 return div;
};
legend.addTo(map);

var controller = L.control.layers(baseMaps, overlayMaps);
controller.addTo(map);

baseMaps["By Number of Burger Places"].addTo(map);