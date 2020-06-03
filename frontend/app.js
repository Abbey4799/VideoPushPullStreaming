var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');

var indexRouter = require('./routes/index');
var usersRouter = require('./routes/users');
var AAARouter = require('./routes/AAA');
var BBBRouter = require('./routes/BBB');
var CCCRouter = require('./routes/CCC');
var DDDRouter = require('./routes/DDD');
var EEERouter = require('./routes/EEE');
var FFFRouter = require('./routes/FFF');

var app = express();

//设置端口
app.set('port', process.env.PORT || 1100);


// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'jade');
// app.set('views',path.join(__dirname , 'views') );
// app.engine('.html', require('ejs').__express);  
// app.set('view engine', 'html'); 

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', indexRouter);
app.use('/users', usersRouter);
app.use('/AAA', AAARouter);
app.use('/BBB', BBBRouter);
app.use('/CCC', CCCRouter);
app.use('/DDD', DDDRouter);
app.use('/EEE', EEERouter);
app.use('/FFF', FFFRouter);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

//设置跨域访问
//allow custom header and CORS
app.all('*', function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Content-Type,Content-Length, Authorization, Accept,X-Requested-With");
  res.header("Access-Control-Allow-Methods","PUT,POST,GET,DELETE,OPTIONS");
  res.header("X-Powered-By",' 3.2.1')
  if(req.method=="OPTIONS") res.send(200);/*让options请求快速返回*/
  else  next();
});

app.listen(app.get('port'));

module.exports = app;
