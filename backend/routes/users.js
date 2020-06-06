var express = require('express');
var router = express.Router();

/* GET users listing. */
router.all('/', function(req, res, next) {
  /*
  var data = req.body;
  console.log(data.time);
  data ={
    'operation': 'live'
  }
  res.send({ret:"OK"});
  */
 var data = req.body
 var op = data['operation']
 if(op=='live')
 {
   var exec = require('child_process').exec;
   var arg1 = '1';
   var arg2 = 'testvideo';
   var arg3 = 'none';
   exec('python /home/ubuntu/html/backend/VideoNet/VideoDB.py'+ arg1+' '+arg2+' '+arg3+' ',function(error,stdout,stderr){
       if(stdout.length >1){
           console.log('you offer args:',stdout);
       } 
       else {
           console.log('you don\'t offer args');
       }
       if(error) {
           console.info('stderr : '+stderr);
       }
  });


 }
 else
 {
   var video_class = op
   var exec = require('child_process').exec;
   var arg1 = '2';
   var arg2 = 'none';
   var arg3 = video_class;
   //var VideoArray = [];
   console.log('users..............');
   //chew
   //console.log(video_class);
   exec('python /home/ubuntu/html/backend/VideoNet/VideoDB.py ' +' '+arg1+' '+arg2+' '+arg3,function(error,stdout,stderr){
        //console.log(stdout);
        stdout = stdout.replace(/[\r\n]/g,"");
        //console.log(stdout);
        if(stdout.length >1){
           //console.log('you offer args:',stdout);
           var cnt = 0;
           var first = 0;
           var second = 0;

           for(var i=0;i<stdout.length;i++)
           {
               
               
              if(stdout[i]==',' && cnt==0)
              {
                 video_class = stdout.substring(0,i-1);
                 //console.log(video_class)
                 first = i;
                 cnt += 1;
              }
              else if (stdout[i]==']' && cnt==1)
              {
                  
                  video_addre = stdout.substring(first+2, i);
                  second = i;
                  cnt += 1;
                  //console.log(video_addre)

              }
              else if (stdout[i]==']' && cnt==2)
              {

                  pic = stdout.substring(second+4,i+1);
                  //console.log(pic);
                  time = stdout.substring(i+4);
                  //video_addre = video_addre.replace('/home/ubuntu/html/frontend/public','');
                  video = {'class':video_class,'address':video_addre, 'pic':pic, 'time':time};
                  //VideoArray.push(video);
                  //console.log(video);
                  break;
              }
              
           }
          
           video_class = video['class'];
           //console.log(video_class);
           address = video['address'];
           pic = video['pic'];
           time = video['time'];

           var adds = [];
           var pics = [];
           var times = [];
           var left = 2;
           var right = 0;
           console.log(address);
           for(var i=0;i<address.length;i++)
           {
               if(address[i]==',')
               {
                   //console.log(i);
                   right = i;
                   //console.log(right);
                   var tmp = address.substring(left, right-1);
                   tmp = tmp.replace('/home/ubuntu/html/frontend/public','');
                   adds.push(tmp);
                   //console.log(tmp);
                   left = i+3;
               }

           }
           if (right != 0)
           {
               var tmp2 = address.substring(left,address.length-1);
               tmp2 = tmp2.replace('/home/ubuntu/html/frontend/public','');
               //console.log(tmp2);
               adds.push(tmp2);
               //console.log(adds);
           }
           else{
                var tmp2 = address.substring(left,address.length-1);
                tmp2 = tmp2.replace('/home/ubuntu/html/frontend/public','');
                //console.log(tmp2);
                adds.push(tmp2);
            //console.log(adds);
           }
           




           var left = 2;
           var right = 0;
           //console.log(address);
           for(var i=0;i<pic.length;i++)
           {
               if(pic[i]==',')
               {
                   //console.log(i);
                   right = i;
                   //console.log(right);
                   var tmp = pic.substring(left, right-1);
                   tmp = tmp.replace('/home/ubuntu/html/frontend/public','');
                   pics.push(tmp);
                   //console.log(tmp);
                   left = i+3;
               }

           }
           var tmp2 = pic.substring(left,pic.length-2);
           tmp2 = tmp2.replace('/home/ubuntu/html/frontend/public','');
           //console.log(tmp2);
           pics.push(tmp2);
           //console.log(pics);

           //console.log(time);
           var left = 1;
           var right = 0;
           //console.log(address);
           for(var i=0;i<time.length;i++)
           {
               if(time[i]==',')
               {
                   //console.log(i);
                   right = i;
                   //console.log(right);
                   var tmp = time.substring(left, right);
                   tmp = parseFloat(tmp);
                   times.push(tmp);
                   //console.log(tmp);
                   left = i+2;
               }

           }
           var tmp2 = time.substring(left,time.length-1);
           //console.log(tmp2);
           tmp2 = parseFloat(tmp2);
           times.push(tmp2);
           //console.log(times);
           //console.log(adds);
           //console.log(pic);
           //console.log(time);

           /* send to front end */
           var VideoArray = {'class':video_class, 'address':adds, 'pic':pics, 'time':times};
           
           console.log(VideoArray);

           res.send({ret:VideoArray});
           /*console.log('address:',address);*/
           
       } 
       else {
           console.log('you don\'t offer args');
       }
       if(error) {
           console.info('stderr : '+stderr);
       }
     });

 }

});

module.exports = router;
