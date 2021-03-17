<sld:StyledLayerDescriptor xmlns="http://www.opengis.net/sld" xmlns:sld="http://www.opengis.net/sld" xmlns:ogc="http://www.opengis.net/ogc" xmlns:gml="http://www.opengis.net/gml" version="1.0.0">
  <sld:NamedLayer>
    <sld:Name>Default Styler</sld:Name>
    <sld:UserStyle>
      <sld:Name>Default Styler</sld:Name>
      <sld:FeatureTypeStyle>
             <sld:Name>&lt; 5.0 mm/yr</sld:Name>
    <sld:Rule>
      <sld:Title>&lt; 5.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>5.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1000.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0000c2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
         <sld:Name>&lt; 5.0 mm/yr</sld:Name>
    <sld:Rule>
      <sld:Title>&lt; 5.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>5.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1000.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0000c2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
         <sld:Name>&lt; 5.0 mm/yr</sld:Name>
    <sld:Rule>
      <sld:Title>&lt; 5.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>5.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1000.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0000c2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
         <sld:Name>&lt; 5.0 mm/yr</sld:Name>
    <sld:Rule>
      <sld:Title>&lt; 5.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>5.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1000.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0000c2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>4.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>5.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0021c1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>4.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>5.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0021c1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>4.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>5.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0021c1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>4.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>5.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0021c1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>4.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0143bf</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>4.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0143bf</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>4.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0143bf</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>4.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0143bf</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>3.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0265be</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>3.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0265be</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>3.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0265be</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>3.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>4.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0265be</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>3.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0387bc</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>3.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0387bc</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>3.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0387bc</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>3.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#0387bc</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>2.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#04a8ba</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>2.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#04a8ba</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>2.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#04a8ba</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>2.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>3.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#04a8ba</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>2.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#04b09e</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>2.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#04b09e</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>2.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#04b09e</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>2.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#04b09e</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>1.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#03b37b</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>1.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#03b37b</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>1.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#03b37b</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>1.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>2.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#03b37b</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>1.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#02b559</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>1.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#02b559</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>1.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#02b559</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>1.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#02b559</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>0.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#01b736</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>0.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#01b736</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>0.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#01b736</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>0.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>1.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#01b736</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>0.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#00ba13</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>0.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#00ba13</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>0.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#00ba13</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>0.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#00ba13</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-0.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-0.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#12c004</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-0.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-0.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#12c004</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-0.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-0.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#12c004</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-0.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-0.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>0.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#12c004</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-0.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-0.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#3fca0e</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-0.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-0.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#3fca0e</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-0.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-0.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#3fca0e</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-0.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-0.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#3fca0e</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-1.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#6cd418</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-1.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#6cd418</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-1.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#6cd418</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-1.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#6cd418</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-1.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#98de22</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-1.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#98de22</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-1.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#98de22</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-1.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#98de22</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-2.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#c5e82c</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-2.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#c5e82c</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-2.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#c5e82c</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-2.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#c5e82c</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-2.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e6e431</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-2.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e6e431</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-2.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e6e431</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-2.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-2.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e6e431</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-3.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e5b627</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-3.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e5b627</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-3.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e5b627</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-3.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e5b627</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-3.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e5891d</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-3.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e5891d</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-3.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e5891d</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-3.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-3.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e5891d</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-4.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e45b13</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-4.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e45b13</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-4.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e45b13</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-4.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.5</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e45b13</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-4.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-5.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e32d09</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-4.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-5.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e32d09</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-4.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-5.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e32d09</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>-4.5 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-5.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-4.5</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e32d09</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>&gt; -5.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1000.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-5.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>20000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>200000000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e20000</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">1</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>&gt; -5.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1000.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-5.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>8000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>20000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e20000</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">2</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>&gt; -5.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1000.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-5.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>3000</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>8000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e20000</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">4</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
    <sld:Rule>
      <sld:Title>&gt; -5.0 mm/yr</sld:Title>
     <ogc:Filter>
      <ogc:And>
       <ogc:PropertyIsGreaterThan>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-1000.0</ogc:Literal>
       </ogc:PropertyIsGreaterThan>
       <ogc:PropertyIsLessThanOrEqualTo>
        <ogc:PropertyName>velocity</ogc:PropertyName>
        <ogc:Literal>-5.0</ogc:Literal>
       </ogc:PropertyIsLessThanOrEqualTo>
      </ogc:And>
     </ogc:Filter>
     <sld:MinScaleDenominator>100</sld:MinScaleDenominator>
     <sld:MaxScaleDenominator>3000</sld:MaxScaleDenominator>
          <sld:LineSymbolizer>
            <sld:Stroke>
              <sld:SvgParameter name="stroke">#e20000</sld:SvgParameter>
              <sld:SvgParameter name="stroke-width">8</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linejoin">bevel</sld:SvgParameter>
              <sld:SvgParameter name="stroke-linecap">square</sld:SvgParameter>
            </sld:Stroke>
          </sld:LineSymbolizer>
    </sld:Rule>
      </sld:FeatureTypeStyle>
    </sld:UserStyle>
  </sld:NamedLayer>
</sld:StyledLayerDescriptor>
