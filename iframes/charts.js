var neighborhoods = [];

d3.csv("data/by_neighborhood.csv", function(csv) {
    csv.forEach(function(row) {
        neighborhoods.push(row);
    })

    var trace1 = {
        x: neighborhoods.map(entry => entry.Neighborhood),
        y: neighborhoods.map(entry => entry.Venue),
        type: "bar",
    };

    var trace2 = {
        x: neighborhoods.map(entry => entry.Neighborhood),
        y: neighborhoods.map(entry => entry.Rating),
        type: "bar",
    };

    var trace3 = {
        x: neighborhoods.map(entry => entry.Neighborhood),
        y: neighborhoods.map(entry => entry['Like Count']),
        type: "bar",
    };

    var trace4 = {
        x: neighborhoods.map(entry => entry.Neighborhood),
        y: neighborhoods.map(entry => entry['Tip Count']),
        type: "bar",
    };

    Plotly.plot('chart', [trace1, trace3, trace4, trace2], {
        updatemenus: [{
            x: .5,
            y: 1.15,
            yanchor: 'top',
            buttons: [{
                method: 'restyle',
                args: ['visible', [true, false, false, false]],
                label: 'Number of Venues'
            }, {
                method: 'restyle',
                args: ['visible', [false, true, false, false]],
                label: 'Number of Likes'
            }, {
                method: 'restyle',
                args: ['visible', [false, false, true, false]],
                label: 'Number of Reviews'
            }, {
                method: 'restyle',
                args: ['visible', [false, false, false, true]],
                label: 'Average Rating'
            }]
        }],
    });

    Plotly.restyle('chart', {
        'visible': false
    }, [1, 2, 3]);
});

