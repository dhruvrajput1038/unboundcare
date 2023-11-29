document.addEventListener("DOMContentLoaded", function() {
    // Get references to the anchor tags and target sections
    var link1 = document.getElementById("id1");
    var link2 = document.getElementById("id2");
    var link3 = document.getElementById("id3");

    var section1 = document.getElementById("s1");
    var section2 = document.getElementById("s2");
    var section3 = document.getElementById("s3");

    // Function to scroll to a target section
    function scrollToTarget(section) {
        var scrollToPosition = section.offsetTop;
        window.scrollTo({
            top: scrollToPosition,
            behavior: "smooth"
        });
    }

    // Add click event listeners to the anchor tags
    link1.addEventListener("click", function(event) {
        event.preventDefault(); // Prevent the default anchor behavior
        scrollToTarget(section1);
    });

    link2.addEventListener("click", function(event) {
        event.preventDefault(); // Prevent the default anchor behavior
        scrollToTarget(section2);
    });

    link3.addEventListener("click", function(event) {
        event.preventDefault(); // Prevent the default anchor behavior
        scrollToTarget(section3);
    });
});
