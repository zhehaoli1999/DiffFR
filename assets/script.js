const now = new Date();
const hours = now.getHours();

var light = hours > 6 && hours < 19;

function lightmode () {
	$('html').attr('data-bs-theme', 'light');
	$('#signature').attr('src', 'assets/signature_black.png');
	$('#teaser-img').attr('src', 'figs/teaser.png');
	$('.btn').removeClass('btn-outline-light');
	$('.btn').addClass('btn-outline-dark');
	$('#bibtex').removeClass('bibtex-dark');
	$('#bibtex').addClass('bibtex-light');
}

function darkmode () {
	$('html').attr('data-bs-theme', 'dark');
	$('#signature').attr('src', 'assets/signature_white.png');
	$('#teaser-img').attr('src', 'figs/teaser.png');
	$('.btn').removeClass('btn-outline-dark');
	$('.btn').addClass('btn-outline-light');
	$('#bibtex').removeClass('bibtex-light');
	$('#bibtex').addClass('bibtex-dark');
}

// color mode init
$(document).ready(function() {
	if (light) lightmode();
	else {
		$('#profile-img').addClass('rotated');
		darkmode();
	}
});

// profile animation
$('#profile-img').click(function() {
	$(this).toggleClass('rotated');

	if (light) darkmode();
	else lightmode();

	light = !light;
});

// bibtex
$('.btn-bibtex').click(function() {
	$('#bibtex > code').html($(this).children('.bibtex-content').html());
	$('.btn-download').attr('href', $(this).children('a').attr('href'));
});
$('.btn-copy').click(function() {
	navigator.clipboard.writeText($('#bibtex > code').html());
});
