$(document).ready(function () {
    var flickerAPI = "javascript/lipstick_data.json";
    $.getJSON(flickerAPI, {
        tags: "lipsticks",
        tagmode: "any",
        format: "json"
    })

        .done(function (data) {
            $.each(data.lipstick1, function (i, lipstick1) {
                $('#img1').attr("src", lipstick1.img1_url).appendTo("#img1");
                $('#cancel1').attr("src", lipstick1.cancelimg_url).appendTo("#cancel1");
                $('#name1').append(lipstick1.name1);
                $('#desc1').append(lipstick1.desc1);
                $('#price1').append(lipstick1.price1);
            });
        });
})();