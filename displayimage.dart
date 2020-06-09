import 'package:flutter/material.dart';
import 'dart:io';
import 'package:async/async.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart';
import 'package:toast/toast.dart';

class Displayimage extends StatefulWidget {

  @override
  _DisplayimageState createState() => _DisplayimageState();
}

class _DisplayimageState extends State<Displayimage> {

  uploadImageToServer(File imageFile, _controller, BuildContext context) async
  {
      var ipAddress = _controller.text;
      print("attempting to connect to server……");
      var stream = new http.ByteStream(DelegatingStream.typed(imageFile.openRead()));
      var length = await imageFile.length();
      print(length);
      var uri = Uri.parse('http://'+ipAddress+':5000/upload');
      print("connection established");
      Toast.show('wait for a few seconds', context, gravity: Toast.BOTTOM, duration: Toast.LENGTH_LONG);
      var request = new http.MultipartRequest("POST", uri);
      var multipartFile = new http.MultipartFile('file', stream, length,
        filename: basename(imageFile.path));
      request.files.add(multipartFile);
      var response = await request.send();
      print(response.statusCode);
      final st = await response.stream.bytesToString();
      print(st);
      Navigator.pushNamed(context, '/result',arguments: {
        'last': st
      });
      return 'hi';
  }

  Widget getWidget(File image, _controller, BuildContext  contex){
    if(image == null){
      return Text('No Image Selected',
      style: TextStyle(fontSize: 20.0));
    }
    else{
     return Column(
       children: <Widget>[
         Image.file(image,height: 400.0,width: 350.0),
         TextField(controller: _controller,
          decoration: InputDecoration(
            enabledBorder: OutlineInputBorder(
              borderSide: BorderSide(color: Colors.red),
            ),
            hintText: "server's IP address"
          ),
        ),
         FloatingActionButton(onPressed: () =>{uploadImageToServer(image, _controller, contex)},
         child: Icon(Icons.arrow_upward),)
       ],
     );

    }
  }

  Map data = {};
  @override
  Widget build(BuildContext context) {
    data = ModalRoute.of(context).settings.arguments;
    File image = data['img'];
    final myController = TextEditingController();
    return Scaffold(
      appBar: AppBar(
        title: Text('Display Image'), 
      ),
      body: ListView(
        children: <Widget>[
          getWidget(image, myController, context),
        ] ,
      ),
    );
  }
}