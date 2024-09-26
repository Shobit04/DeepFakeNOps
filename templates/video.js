$(document).ready(function() {
    $('#videoForm').submit(function(event) {
        event.preventDefault();
        var videoPath = $('#videoPath').val();
        predict(videoPath);
    });

    function predict(videoPath) {
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: { videoPath: videoPath },
            success: function(response) {
                $('#predictionResult').text(response);
            },
            error: function(xhr, status, error) {
                console.error(error);
            }
        });
    }
});
