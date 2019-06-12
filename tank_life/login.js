var attempt = 3; // Variable to count number of attempts.
// Below function Executes on click of login button.
function validate(){
var username = document.getElementById("username").value;
var password = document.getElementById("password").value;
if ( username == "kiran" && password == "kiran"){
window.location = "index.html"; // Redirecting to other page.

document.getElementById("username").value="";
return false;
}
else{
attempt --;// Decrementing by one.
alert("WRONG USERNAME OR PASSWORD")
alert("You are left with "+attempt+" attempts;");
// Disabling fields after 3 attempts.
if( attempt == 0){
document.getElementById("username").disabled = true;
document.getElementById("password").disabled = true;
document.getElementById("submit").disabled = true;
return false;
}
}
}