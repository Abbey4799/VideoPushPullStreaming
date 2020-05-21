var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  //注意从此要加后缀了
  res.render('index.ejs', { title: 'Express' });
});

module.exports = router;
