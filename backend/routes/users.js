var express = require('express');
var router = express.Router();

/* GET users listing. */
router.all('/', function(req, res, next) {
  var data = req.body;
  console.log(data.time);
  res.send({ret:"OK"});
});

module.exports = router;
